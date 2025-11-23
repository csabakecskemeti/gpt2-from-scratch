import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass

from model import GPT


class GPT2Inference:
    """ To generate text sequences using a trained GPT2 model """

    def __init__(self, model, token_encoder, device):
        self.model = model
        self.token_encoder = token_encoder
        self.device = device
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    def generate_sequences(self, prompt, num_seq=5, max_tokens=50):
        self.model.eval()
        tokens = self.token_encoder.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)    # (n,)   n : current sequence length
        tokens = tokens.unsqueeze(0).repeat(num_seq, 1)    # (1,n) --> (num_seq, n)
        gen_tokens = tokens.to(self.device)
        # create a different rng generator so as not to impact the global rng state used for training
        sample_rng = torch.Generator(device=self.device).manual_seed(42)

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
        for i in range(num_seq):
            tokens = gen_tokens[i, :max_tokens].tolist()
            gen_text = self.token_encoder.decode(tokens)
            print(f"> sample {i}: {gen_text}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Hello, I am a language model,")
    parser.add_argument('--num_seq', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=50)
    args = parser.parse_args()
    return args


@dataclass
class GPTConfig:
    context_length: int = 1024    # max context / sequence length
    vocab_size: int = 50257    # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    num_layers: int = 12
    embd_size: int = 768    # embedding dim
    num_heads: int = 12


def inference(args=None):
    if args is None:
        args = parse_args()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'    # for apple macbook GPUs
    print(f'using device: {device}')

    model_path = './checkpoints/best_model.pt'
    checkpoint = torch.load(model_path, weights_only=False)
    print(f"loaded model from: {model_path}")
    
    # Handle both old checkpoints (without 'config') and new ones (with 'config')
    if 'config' in checkpoint:
        # New checkpoint format - use saved config
        config = checkpoint['config']
        print("✓ Loaded model config from checkpoint")
    elif 'args' in checkpoint:
        # Old checkpoint format - reconstruct config from args
        print("⚠️  Old checkpoint format detected, reconstructing config from args")
        ckpt_args = checkpoint['args']
        config = GPTConfig(
            vocab_size=50304,  # Same as training
            context_length=ckpt_args.get('context_length', 1024),
            num_layers=ckpt_args.get('num_layers', 12),
            num_heads=ckpt_args.get('num_heads', 12),
            embd_size=ckpt_args.get('embd_size', 768)
        )
    else:
        # Fallback - use default config
        print("⚠️  No config or args found in checkpoint, using default GPTConfig")
        config = GPTConfig()
    
    print(f"Model config: vocab_size={config.vocab_size}, context_length={config.context_length}, "
          f"num_layers={config.num_layers}, num_heads={config.num_heads}, embd_size={config.embd_size}")

    model = GPT(config=config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    token_encoder = tiktoken.get_encoding('gpt2')
    generator = GPT2Inference(model, token_encoder, device)

    generator.generate_sequences(args.prompt, args.num_seq, args.max_tokens)


if __name__ == '__main__':
    inference()
