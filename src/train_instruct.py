"""
Instruction Fine-Tuning Script for GPT-2
Fine-tunes a pre-trained GPT-2 model on instruction-response pairs.
"""

import os
import math
import numpy as np
import time
import random
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from model import GPT, GPTConfig
from dataloader_instruct import InstructDataLoader

torch.set_float32_matmul_precision('high')


class CheckpointManager:
    """Checkpoint management for instruction fine-tuning"""
    
    def __init__(self, checkpoint_dir, master_process=True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.master_process = master_process
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, step, model, optimizer, train_loss, val_loss, 
                       train_loader, val_loader, args_dict):
        """Save checkpoint"""
        if not self.master_process:
            return
        
        # Collect RNG states
        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().cpu(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        if rng_state['torch_cuda'] is not None:
            rng_state['torch_cuda'] = [s.cpu() for s in rng_state['torch_cuda']]
        
        # Collect dataloader states
        dataloader_state = {
            'train': train_loader.get_state(),
            'val': val_loader.get_state(),
        }
        
        checkpoint = {
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': model.config,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'rng_state': rng_state,
            'dataloader_state': dataloader_state,
            'args': args_dict,
            'timestamp': time.time(),
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        print(f"💾 Saved checkpoint: {latest_path}")
        
        # Save step checkpoint every 500 steps
        if step % 500 == 0:
            step_path = self.checkpoint_dir / f'step_{step:06d}.pt'
            torch.save(checkpoint, step_path)
            print(f"💾 Saved step checkpoint: {step_path}")
    
    def load_checkpoint(self, checkpoint_path, model, optimizer, train_loader, val_loader):
        """Load checkpoint and restore state"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load model and optimizer
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Restore RNG states
        random.setstate(checkpoint['rng_state']['python'])
        np.random.set_state(checkpoint['rng_state']['numpy'])
        
        torch_rng_state = checkpoint['rng_state']['torch']
        if isinstance(torch_rng_state, torch.Tensor):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        
        if checkpoint['rng_state']['torch_cuda'] is not None and torch.cuda.is_available():
            cuda_rng_states = checkpoint['rng_state']['torch_cuda']
            if isinstance(cuda_rng_states, list):
                cuda_rng_states = [s.cpu() if isinstance(s, torch.Tensor) and s.device.type == 'cuda' else s 
                                 for s in cuda_rng_states]
            torch.cuda.set_rng_state_all(cuda_rng_states)
        
        # Restore dataloader states
        train_loader.set_state(checkpoint['dataloader_state']['train'])
        val_loader.set_state(checkpoint['dataloader_state']['val'])
        
        step = checkpoint['step']
        print(f"✓ Resumed from step {step}")
        
        return step


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with warmup"""
    # Warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # Cosine decay
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Instruction Fine-Tuning")
    
    # Model args (loaded from checkpoint)
    parser.add_argument('--pretrained_model', type=str, default='checkpoints/best_model.pt',
                       help='Path to pre-trained model checkpoint')
    
    # Training args
    parser.add_argument('--max_steps', type=int, default=2000, help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Peak learning rate')
    parser.add_argument('--min_lr', type=float, default=2e-6, help='Minimum learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Batch size args
    parser.add_argument('--mini_batch_size', type=int, default=8, help='Mini batch size per GPU')
    parser.add_argument('--total_batch_size', type=int, default=65536, help='Total batch size in tokens')
    parser.add_argument('--context_length', type=int, default=1024, help='Context length')
    
    # Checkpoint args
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_instruct',
                       help='Directory for checkpoints')
    parser.add_argument('--eval_freq', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--checkpoint_freq', type=int, default=100, help='Checkpoint frequency')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (path or "latest")')
    
    # TensorBoard
    parser.add_argument('--use_tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--tensorboard_dir', type=str, default='runs_instruct',
                       help='TensorBoard log directory')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data_instruct',
                       help='Directory containing instruction data')
    
    args = parser.parse_args()
    return args


def main():
    # Get arguments
    args = get_args()
    
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA required for DDP"
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if master_process:
        print("="*80)
        print("Instruction Fine-Tuning for GPT-2")
        print("="*80)
        print(f"Pre-trained model: {args.pretrained_model}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Max steps: {args.max_steps}")
        print(f"Batch size: {args.mini_batch_size} per GPU, {args.total_batch_size} total")
        print(f"Checkpoint dir: {args.checkpoint_dir}")
        print(f"Device: {device}")
        print(f"DDP: {ddp} (world size: {ddp_world_size})")
        print("="*80)
        print()
    
    # Calculate gradient accumulation steps
    B = args.mini_batch_size
    T = args.context_length
    assert args.total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = args.total_batch_size // (B * T * ddp_world_size)
    
    if master_process:
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print()
    
    # Set random seed
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    # Load pre-trained model
    if master_process:
        print(f"Loading pre-trained model from {args.pretrained_model}...")
    
    pretrained_ckpt = torch.load(args.pretrained_model, map_location='cpu', weights_only=False)
    
    # Get config from checkpoint or create default
    if 'config' in pretrained_ckpt:
        config = pretrained_ckpt['config']
    else:
        # Reconstruct config from args if not in checkpoint
        ckpt_args = pretrained_ckpt.get('args', {})
        config = GPTConfig(
            vocab_size=50304,
            context_length=ckpt_args.get('context_length', 1024),
            num_layers=ckpt_args.get('num_layers', 12),
            num_heads=ckpt_args.get('num_heads', 12),
            embd_size=ckpt_args.get('embd_size', 768)
        )
    
    model = GPT(config=config)
    model.load_state_dict(pretrained_ckpt['model'])
    model.to(device)
    
    if master_process:
        print(f"✓ Loaded pre-trained model")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print()
    
    # Wrap model in DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    
    # Create dataloaders
    train_loader = InstructDataLoader(
        B=B, T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split='train',
        data_dir=args.data_dir
    )
    
    val_loader = InstructDataLoader(
        B=B, T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split='val',
        data_dir=args.data_dir
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    # Setup checkpointing
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        master_process=master_process
    )
    
    # Setup TensorBoard
    writer = None
    if args.use_tensorboard and master_process:
        from datetime import datetime
        run_name = datetime.now().strftime('instruct_%Y%m%d-%H%M%S')
        writer = SummaryWriter(log_dir=f"{args.tensorboard_dir}/{run_name}")
        print(f"📊 TensorBoard logging to: {args.tensorboard_dir}/{run_name}")
        print()
    
    # Resume if requested
    start_step = 0
    if args.resume:
        if args.resume == 'latest':
            checkpoint_path = Path(args.checkpoint_dir) / 'latest.pt'
        else:
            checkpoint_path = Path(args.resume)
        
        if checkpoint_path.exists():
            start_step = checkpoint_manager.load_checkpoint(
                checkpoint_path, raw_model, optimizer, train_loader, val_loader
            )
            start_step += 1  # Start from next step
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print("Starting from scratch...")
    
    # Training loop
    if master_process:
        print("\n" + "="*80)
        print("Starting Instruction Fine-Tuning")
        print("="*80)
        print()
    
    model.train()
    for step in range(start_step, args.max_steps):
        t0 = time.time()
        
        # Get learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training step
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
            
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            loss.backward()
        
        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Synchronize loss across all processes
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        # Timing
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt
        
        # Logging
        if master_process and step % 10 == 0:
            print(f"step {step:5d} | loss {loss_accum.item():.4f} | lr {lr:.2e} | "
                  f"norm {norm:.4f} | dt {dt*1000:.0f}ms | tok/sec {tokens_per_sec:.0f}")
        
        # TensorBoard logging
        if writer and step % 10 == 0:
            writer.add_scalar('train/loss', loss_accum.item(), step)
            writer.add_scalar('train/learning_rate', lr, step)
            writer.add_scalar('train/gradient_norm', norm, step)
            writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)
        
        # Evaluation
        if step % args.eval_freq == 0:
            model.eval()
            val_loss_accum = 0.0
            val_steps = 20
            
            for _ in range(val_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    val_loss_accum += loss.detach()
            
            val_loss = val_loss_accum / val_steps
            
            if ddp:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            
            if master_process:
                print(f"\n{'='*80}")
                print(f"Validation at step {step}: loss = {val_loss.item():.4f}")
                print(f"{'='*80}\n")
            
            if writer:
                writer.add_scalar('val/loss', val_loss.item(), step)
            
            model.train()
        
        # Checkpointing
        if step % args.checkpoint_freq == 0 and step > 0:
            checkpoint_manager.save_checkpoint(
                step=step,
                model=raw_model,
                optimizer=optimizer,
                train_loss=loss_accum.item(),
                val_loss=0.0,
                train_loader=train_loader,
                val_loader=val_loader,
                args_dict=vars(args)
            )
    
    # Final checkpoint
    if master_process:
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        checkpoint_manager.save_checkpoint(
            step=args.max_steps,
            model=raw_model,
            optimizer=optimizer,
            train_loss=loss_accum.item(),
            val_loss=0.0,
            train_loader=train_loader,
            val_loader=val_loader,
            args_dict=vars(args)
        )
    
    if ddp:
        dist.destroy_process_group()
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()

