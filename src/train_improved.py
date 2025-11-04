import os
import math
import numpy as np
import time
import random
import signal
import sys
import glob
from dataclasses import dataclass
from pathlib import Path
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from model import GPT
from dataloader import DataLoaderLite
from hellaswag_eval import render_example, iterate_examples, get_most_likely_row

torch.set_float32_matmul_precision('high')    # enable TF32 precision

# set torch compile to True (if it doesn't throws any error) to speed up training
use_torch_compile = False


class CheckpointManager:
    """
    Manages 3-tier checkpointing system:
    1. Latest checkpoint (always overwrite for quick resume)
    2. Rolling checkpoints (keep last N, default 10)
    3. Epoch checkpoints (keep all)
    """
    def __init__(self, checkpoint_dir, keep_last_n=10, master_process=True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.master_process = master_process
        
    def save_checkpoint(self, 
                       step, 
                       epoch, 
                       model, 
                       optimizer, 
                       train_loss, 
                       val_loss,
                       train_loader,
                       val_loader,
                       args_dict,
                       is_best=False,
                       is_epoch_end=False):
        """Save a complete training checkpoint"""
        if not self.master_process:
            return  # Only master process saves checkpoints
            
        # Gather RNG states for reproducibility
        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        
        # Get data loader state (current shard and position)
        dataloader_state = {
            'train_curr_shard': train_loader.curr_shard,
            'train_curr_pos': train_loader.curr_pos,
            'val_curr_shard': val_loader.curr_shard,
            'val_curr_pos': val_loader.curr_pos,
        }
        
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'rng_state': rng_state,
            'dataloader_state': dataloader_state,
            'args': args_dict,
            'timestamp': time.time(),
        }
        
        # 1. Always save as latest.pt (for quick resume)
        latest_path = self.checkpoint_dir / 'latest.pt'
        latest_backup_path = self.checkpoint_dir / 'latest_backup.pt'
        
        # Backup previous latest before overwriting
        if latest_path.exists():
            if latest_backup_path.exists():
                latest_backup_path.unlink()
            latest_path.rename(latest_backup_path)
        
        torch.save(checkpoint, latest_path)
        print(f'✓ Saved latest checkpoint: {latest_path}')
        
        # 2. Save as rolling checkpoint (keep last N)
        rolling_path = self.checkpoint_dir / f'rolling_step_{step:06d}.pt'
        torch.save(checkpoint, rolling_path)
        print(f'✓ Saved rolling checkpoint: {rolling_path}')
        
        # Cleanup old rolling checkpoints (keep only last N)
        self._cleanup_rolling_checkpoints()
        
        # 3. Save epoch checkpoint (keep all)
        if is_epoch_end:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch:05d}.pt'
            torch.save(checkpoint, epoch_path)
            print(f'✓ Saved epoch checkpoint: {epoch_path}')
        
        # 4. Save best model if this is the best so far
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f'🏆 Saved best model checkpoint: {best_path}')
    
    def _cleanup_rolling_checkpoints(self):
        """Keep only the last N rolling checkpoints"""
        rolling_checkpoints = sorted(self.checkpoint_dir.glob('rolling_step_*.pt'))
        
        if len(rolling_checkpoints) > self.keep_last_n:
            # Remove oldest checkpoints
            for ckpt in rolling_checkpoints[:-self.keep_last_n]:
                ckpt.unlink()
                print(f'  Cleaned up old checkpoint: {ckpt.name}')
    
    def load_checkpoint(self, checkpoint_path, model, optimizer, train_loader, val_loader, device):
        """Load a checkpoint and restore complete training state"""
        if checkpoint_path == 'latest':
            checkpoint_path = self.checkpoint_dir / 'latest.pt'
        elif checkpoint_path == 'best':
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\n{'='*60}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        print(f"{'='*60}")
        
        # Load checkpoint
        # Note: weights_only=False is required for checkpoints containing numpy arrays
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Restore model and optimizer
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Restore RNG states for reproducibility
        random.setstate(checkpoint['rng_state']['python'])
        np.random.set_state(checkpoint['rng_state']['numpy'])
        
        # Ensure torch RNG state is on CPU (torch.set_rng_state requires CPU tensors)
        torch_rng_state = checkpoint['rng_state']['torch']
        if isinstance(torch_rng_state, torch.Tensor):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        
        if checkpoint['rng_state']['torch_cuda'] is not None and torch.cuda.is_available():
            cuda_rng_states = checkpoint['rng_state']['torch_cuda']
            # Ensure CUDA RNG states are on correct device
            if isinstance(cuda_rng_states, list):
                cuda_rng_states = [s.cpu() if isinstance(s, torch.Tensor) and s.device.type == 'cuda' else s for s in cuda_rng_states]
            torch.cuda.set_rng_state_all(cuda_rng_states)
        
        # Restore data loader state
        dataloader_state = checkpoint['dataloader_state']
        train_loader.curr_shard = dataloader_state['train_curr_shard']
        train_loader.curr_pos = dataloader_state['train_curr_pos']
        val_loader.curr_shard = dataloader_state['val_curr_shard']
        val_loader.curr_pos = dataloader_state['val_curr_pos']
        
        # Reload the current shard for both loaders
        train_loader.tokens = train_loader.load_tokens(train_loader.shard_filepaths[train_loader.curr_shard])
        val_loader.tokens = val_loader.load_tokens(val_loader.shard_filepaths[val_loader.curr_shard])
        
        print(f"✓ Restored model and optimizer state")
        print(f"✓ Restored RNG states")
        print(f"✓ Restored data loader state")
        print(f"  - Training: shard {train_loader.curr_shard}, position {train_loader.curr_pos}")
        print(f"  - Validation: shard {val_loader.curr_shard}, position {val_loader.curr_pos}")
        print(f"\nResuming from:")
        print(f"  - Step: {checkpoint['step']}")
        print(f"  - Epoch: {checkpoint['epoch']}")
        print(f"  - Train Loss: {checkpoint['train_loss']:.6f}")
        print(f"  - Val Loss: {checkpoint['val_loss']:.6f}")
        print(f"{'='*60}\n")
        
        return checkpoint['step'], checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        print("\nAvailable checkpoints:")
        print("-" * 60)
        
        # Latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        if latest_path.exists():
            ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
            print(f"  latest: step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
        
        # Best checkpoint
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
            print(f"  best:   step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
        
        # Epoch checkpoints
        epoch_ckpts = sorted(self.checkpoint_dir.glob('epoch_*.pt'))
        if epoch_ckpts:
            print("\nEpoch checkpoints:")
            for path in epoch_ckpts:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                print(f"  {path.name}: step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
        
        # Rolling checkpoints
        rolling_ckpts = sorted(self.checkpoint_dir.glob('rolling_step_*.pt'))
        if rolling_ckpts:
            print("\nRolling checkpoints (last 10):")
            for path in rolling_ckpts[-10:]:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                print(f"  {path.name}: step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
        
        print("-" * 60)


class Trainer:
    def __init__(
            self, 
            model, 
            optimizer, 
            train_loader, 
            val_loader, 
            token_encoder, 
            eval_freq, 
            checkpoint_freq,
            grad_accum_steps, 
            ddp, 
            ddp_rank, 
            ddp_world_size, 
            device, 
            logpath,
            checkpoint_manager,
            args_dict,
            tensorboard_writer=None
    ):
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.master_process = ddp_rank == 0
        self.ddp_world_size = ddp_world_size

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.token_encoder = token_encoder

        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.grad_accum_steps = grad_accum_steps
        self.device = device
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        self.logpath = logpath
        self.checkpoint_manager = checkpoint_manager
        self.args_dict = args_dict
        self.writer = tensorboard_writer  # TensorBoard writer
        
        # Track best validation loss for saving best model
        self.best_val_loss = float('inf')
        
        # Track current epoch
        self.current_epoch = 0
        
        # Handle graceful shutdown on Ctrl+C
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully by saving checkpoint before exit"""
        if self.master_process:
            print("\n" + "="*60)
            print("⚠️  Interrupt signal received! Saving checkpoint before exit...")
            print("="*60)
        self.interrupted = True

    def train(
        self, 
        max_steps, 
        warmup_steps, 
        max_lr, 
        min_lr,
        start_step=0,
        start_epoch=0,
        steps_per_epoch=None
    ):
        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = max_steps // 5  # Assume 5 epochs by default
        
        self.current_epoch = start_epoch
        current_step_in_epoch = start_step % steps_per_epoch if steps_per_epoch > 0 else 0
        
        for step in range(start_step, max_steps):
            # Check for interruption
            if self.interrupted:
                if self.master_process:
                    print("\nSaving emergency checkpoint...")
                    raw_model = self.model.module if self.ddp else self.model
                    self.checkpoint_manager.save_checkpoint(
                        step=step,
                        epoch=self.current_epoch,
                        model=raw_model,
                        optimizer=self.optimizer,
                        train_loss=0.0,  # We don't have current loss
                        val_loss=self.best_val_loss,
                        train_loader=self.train_loader,
                        val_loader=self.val_loader,
                        args_dict=self.args_dict,
                    )
                    print("✓ Emergency checkpoint saved. Exiting...")
                if self.ddp:
                    dist.destroy_process_group()
                sys.exit(0)
            
            # Track epoch boundaries
            if current_step_in_epoch >= steps_per_epoch and steps_per_epoch > 0:
                self.current_epoch += 1
                current_step_in_epoch = 0
                if self.master_process:
                    print(f"\n{'='*60}")
                    print(f"🎯 Completed Epoch {self.current_epoch - 1}")
                    print(f"{'='*60}\n")
            
            t0 = time.time()
            self.is_last_step = (step == max_steps - 1)

            # evaluate validation loss
            val_loss = None
            if step % self.eval_freq == 0 or self.is_last_step:
                val_loss = self.evaluate_validation(step)

            # evaluate model performance on HellaSwag every once in a while
            if ((step > 0 and step % self.eval_freq == 0) or self.is_last_step) and (not use_torch_compile):
                self.evaluate_helloswag(step)

            # generate sequences from the model every once in a while
            if ((step > 0 and step % self.eval_freq == 0) or self.is_last_step) and (not use_torch_compile):
                generated_texts = self.generate_sequences(num_seq=5, max_tokens=32)
                # Log generated text to TensorBoard
                if self.master_process and self.writer is not None:
                    text_samples = "\n\n".join([f"Sample {i+1}:\n{text}" for i, text in enumerate(generated_texts)])
                    self.writer.add_text('Generated_Samples/text', text_samples, step)

            # training loop starts here
            self.model.train()    # sets model to train mode
            self.optimizer.zero_grad()    # resets all gradients
            batch_loss = 0.0
            
            for mini_step in range(self.grad_accum_steps):
                inp, tar = self.train_loader.next_batch()
                inp, tar = inp.to(self.device), tar.to(self.device)
                
                # FORWARD PASS !!!
                # autocast to bfloat16 for faster compute and memory efficiency
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(inp, tar)

                # loss is scaled to account for gradient accumulation
                loss /= self.grad_accum_steps
                batch_loss += loss.detach()

                if self.ddp:
                    # in the final mini_step, sync and avg all gradients across all processes
                    self.model.require_backward_grad_sync = (mini_step == self.grad_accum_steps - 1)

                loss.backward()

            if self.ddp:
                # average out 'batch_loss' across all processes
                dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)

            # once gradients are computed, clip the global l2-norm of the gradient at 1.0
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # determine learning rate with decay
            lr = self.estimate_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            # set learning rate for this iteration
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            self.optimizer.step()
            if self.device_type == 'cuda':
                torch.cuda.synchronize()    # wait for the GPU to finish work
            
            dt = (time.time() - t0) * 1000.0    # in ms
            tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps * self.ddp_world_size
            tokens_per_sec = tokens_processed / dt

            if self.master_process:
                print(f'step {step:5d} | epoch {self.current_epoch} | loss: {batch_loss.item():.6f} | lr: {lr:.2e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}')
                with open(self.logpath, 'a') as f:
                    f.write(f'{step} train {batch_loss.item():.6f}\n')
                
                # TensorBoard logging
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', batch_loss.item(), step)
                    self.writer.add_scalar('Learning_Rate/lr', lr, step)
                    self.writer.add_scalar('Gradient/norm', norm, step)
                    self.writer.add_scalar('Performance/step_time_ms', dt, step)
                    self.writer.add_scalar('Performance/tokens_per_sec', tokens_per_sec, step)
                    self.writer.add_scalar('Training/epoch', self.current_epoch, step)
                    
                    # Log histograms less frequently (every 1000 steps) to save space
                    if step % 1000 == 0:
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                self.writer.add_histogram(f'Gradients/{name}', param.grad, step)
                            self.writer.add_histogram(f'Parameters/{name}', param, step)
            
            # CHECKPOINTING: Save checkpoint periodically
            is_epoch_end = (current_step_in_epoch == steps_per_epoch - 1) and steps_per_epoch > 0
            
            if step > 0 and (step % self.checkpoint_freq == 0 or self.is_last_step or is_epoch_end):
                if self.master_process:
                    raw_model = self.model.module if self.ddp else self.model
                    
                    # Check if this is the best model so far
                    is_best = False
                    if val_loss is not None and val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        is_best = True
                    
                    self.checkpoint_manager.save_checkpoint(
                        step=step,
                        epoch=self.current_epoch,
                        model=raw_model,
                        optimizer=self.optimizer,
                        train_loss=batch_loss.item(),
                        val_loss=val_loss if val_loss is not None else self.best_val_loss,
                        train_loader=self.train_loader,
                        val_loader=self.val_loader,
                        args_dict=self.args_dict,
                        is_best=is_best,
                        is_epoch_end=is_epoch_end
                    )
            
            current_step_in_epoch += 1


    def evaluate_validation(self, step):
        self.model.eval()    # sets model to eval mode
        self.val_loader.reset()
        # evaluate the model on validation set
        with torch.no_grad():
            val_loss_accum = 0.0
            val_steps = 20
            for _ in range(val_steps):
                inp, tar = self.val_loader.next_batch()
                inp, tar = inp.to(self.device), tar.to(self.device)
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(inp, tar)
                loss /= val_steps
                val_loss_accum += loss.detach()

        if self.ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        
        val_loss = val_loss_accum.item()
        
        if self.master_process:
            print(f'Val loss: {val_loss:.4f}')
            with open(self.logpath, 'a') as f:
                f.write(f'{step} val {val_loss:.4f}\n')
            
            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar('Loss/validation', val_loss, step)
        
        return val_loss


    def evaluate_helloswag(self, step):
        """ 
        Construct a batch of 4 sequences and perform token completion using 
        our model. 
        """
        n_total = 0
        n_correct_norm = 0
        for i, example in enumerate(iterate_examples('val')):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % self.ddp_world_size != self.ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)    # (4,N), (4,N), (4,N)
            tokens, mask = tokens.to(self.device), mask.to(self.device)
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            n_total += 1
            n_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if self.ddp:
            n_total = torch.tensor(n_total, device=self.device, dtype=torch.long)
            n_correct_norm = torch.tensor(n_correct_norm, device=self.device, dtype=torch.long)
            dist.all_reduce(n_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_correct_norm, op=dist.ReduceOp.SUM)
            n_total = n_total.item()
            n_correct_norm = n_correct_norm.item()
        acc_norm = n_correct_norm / n_total
        if self.master_process:
            print(f'HelloSwag accuracy: {n_correct_norm}/{n_total}={acc_norm:.4f}')
            with open(self.logpath, 'a') as f:
                f.write(f'{step} hellaswag {acc_norm:.4f}\n')
            
            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar('Evaluation/hellaswag_accuracy', acc_norm, step)


    def generate_sequences(self, num_seq=4, max_tokens=32):
        self.model.eval()
        tokens = self.token_encoder.encode("Hello, I am a language model")
        tokens = torch.tensor(tokens, dtype=torch.long)    # (n,)   n : current sequence length
        tokens = tokens.unsqueeze(0).repeat(num_seq, 1)    # (1,n) --> (num_seq, n)
        gen_tokens = tokens.to(self.device)
        # create a different rng generator so as not to impact the global rng state used for training
        sample_rng = torch.Generator(device=self.device)
        # adding 'ddp_rank' in seeding to generate different tokens for different rank processes
        sample_rng.manual_seed(42 + self.ddp_rank)
        # generate new tokens one token at a time until the sequence length becomes 'max_tokens'
        while gen_tokens.shape[-1] <= max_tokens:
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(gen_tokens)    # (num_seq, n, vocab_size)
                logits = logits[:, -1, :]    # (num_seq, vocab_size)
                probs = F.softmax(logits, dim=-1)    # (num_seq, vocab_size)
                # take top-k 50 probs
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # (num_seq, 50), (num_seq, 50)
                # sample a token from top-50 probabilities
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)    # (num_seq, 1)
                next_tok = torch.gather(topk_indices, -1, ix)    # (num_seq, 1)
                gen_tokens = torch.cat([gen_tokens, next_tok], dim=1)
        # decode generated tokens and print generated text
        generated_texts = []
        for i in range(num_seq):
            tokens = gen_tokens[i, :max_tokens].tolist()
            gen_text = self.token_encoder.decode(tokens)
            print(f"> rank {self.ddp_rank} sample {i}: {gen_text}")
            generated_texts.append(gen_text)
        
        return generated_texts


    def estimate_lr(self, step, warmup_steps, max_steps, max_lr, min_lr):
        """
        Learning rate scheduler: Cosine-decay learning schedule with warmup
        """
        # 1) linear warmup for 'warmup_iters' steps
        if step < warmup_steps:
            return max_lr * (step+1) / warmup_steps
        # 2) if step > lr_decay_iters, return min lr
        if step > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)


@dataclass
class GPTConfig:
    context_length: int = 1024    # max context / sequence length
    vocab_size: int = 50257    # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    num_layers: int = 12
    embd_size: int = 768    # embedding dim
    num_heads: int = 12


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter Configuration")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="number of tokens processed for each weight update")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="setting of mini_batch_size is just a performance optimization")
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--embd_size", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-3 * 0.1)
    parser.add_argument("--warmup_steps", type=int, default=715)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=19073)
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--checkpoint_freq", type=int, default=250, help="save checkpoint every N steps")
    parser.add_argument("--keep_checkpoints", type=int, default=10, help="number of rolling checkpoints to keep")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--logdir", type=str, default="./logs/")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/", help="directory to save checkpoints")
    
    # Resume training arguments
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from (or 'latest'/'best')")
    parser.add_argument("--list_checkpoints", action='store_true', help="list all available checkpoints and exit")
    
    # TensorBoard arguments
    parser.add_argument("--use_tensorboard", action='store_true', help="enable TensorBoard logging")
    parser.add_argument("--tensorboard_dir", type=str, default="./runs/", help="directory for TensorBoard logs")
    parser.add_argument("--run_name", type=str, default=None, help="name for this training run (for TensorBoard)")
    
    return parser.parse_args()


def main():
    args = get_args()

    # Print the hyperparameters
    print("\n" + "="*60)
    print("Hyperparameter Configuration:")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key:20s}: {value}")
    print("="*60 + "\n")

    # set up DDP (distributed data parallel)
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), f'use of DDP requires CUDA'
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        print(f'using device: {device}')

    device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_last_n=args.keep_checkpoints,
        master_process=master_process
    )
    
    # List checkpoints and exit if requested
    if args.list_checkpoints:
        if master_process:
            checkpoint_manager.list_checkpoints()
        sys.exit(0)

    # setting seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # create the logs directory if it doesn't exist
    os.makedirs(args.logdir, exist_ok=True)
    logpath = os.path.join(args.logdir, 'log.txt')
    
    # Don't clear log file if resuming
    if not args.resume:
        with open(logpath, 'w') as f:
            pass

    assert args.total_batch_size % (args.mini_batch_size * args.context_length * ddp_world_size) == 0
    grad_accum_steps = args.total_batch_size // (args.mini_batch_size * args.context_length * ddp_world_size)
    if master_process:
        print(f'desired batch size (number of tokens): {args.total_batch_size}')
        print(f'gradient accumulation steps: {grad_accum_steps}')
    print(f'GPU: {ddp_rank}, {ddp_local_rank}')

    train_loader = DataLoaderLite(B=args.mini_batch_size, T=args.context_length, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B=args.mini_batch_size, T=args.context_length, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

    # create GPT model
    gpt_config = GPTConfig(vocab_size=50304,
                           context_length=args.context_length, 
                           num_layers=args.num_layers, 
                           num_heads=args.num_heads, 
                           embd_size=args.embd_size
                           )
    model = GPT(config=gpt_config)
    model.to(device)
    
    if use_torch_compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, lr=args.max_lr, device_type=device_type, master_process=master_process)
    token_encoder = tiktoken.get_encoding('gpt2')

    # Resume from checkpoint if requested
    start_step = 0
    start_epoch = 0
    if args.resume:
        start_step, start_epoch, _, _ = checkpoint_manager.load_checkpoint(
            checkpoint_path=args.resume,
            model=raw_model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        start_step += 1  # Resume from next step

    start_time = time.time()
    
    # Save args as dictionary for checkpointing
    args_dict = vars(args)
    
    # Initialize TensorBoard writer (only on master process)
    writer = None
    if args.use_tensorboard and master_process:
        # Create run name if not provided
        if args.run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"gpt2_train_{timestamp}"
        else:
            run_name = args.run_name
        
        tensorboard_path = os.path.join(args.tensorboard_dir, run_name)
        writer = SummaryWriter(log_dir=tensorboard_path)
        print(f"\n{'='*60}")
        print(f"TensorBoard logging enabled!")
        print(f"Log directory: {tensorboard_path}")
        print(f"To view: tensorboard --logdir={args.tensorboard_dir}")
        print(f"{'='*60}\n")
        
        # Log hyperparameters
        writer.add_hparams(
            {k: v for k, v in args_dict.items() if isinstance(v, (int, float, str, bool))},
            {'hparams/placeholder': 0}  # Placeholder metric
        )
    
    # init the trainer object
    trainer = Trainer(
        model, 
        optimizer, 
        train_loader, 
        val_loader, 
        token_encoder, 
        args.eval_freq,
        args.checkpoint_freq,
        grad_accum_steps, 
        ddp, 
        ddp_rank, 
        ddp_world_size, 
        device, 
        logpath,
        checkpoint_manager,
        args_dict,
        tensorboard_writer=writer
    )

    max_steps = args.steps_per_epoch * args.num_epochs
    
    if master_process:
        print(f"\n{'='*60}")
        print(f"Starting training from step {start_step} to {max_steps}")
        print(f"{'='*60}\n")
    
    trainer.train(
        max_steps, 
        args.warmup_steps, 
        args.max_lr, 
        args.min_lr,
        start_step=start_step,
        start_epoch=start_epoch,
        steps_per_epoch=args.steps_per_epoch
    )

    dt = (time.time() - start_time) / (60*60)
    if master_process:
        print(f"\n{'='*60}")
        print(f"Total training time: {dt:.4f} hours")
        print(f"{'='*60}\n")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        if master_process:
            print("TensorBoard writer closed.")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

