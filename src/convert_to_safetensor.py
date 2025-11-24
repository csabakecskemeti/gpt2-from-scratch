"""
Simple, direct conversion from PyTorch checkpoint to Hugging Face safetensors.
This version focuses on correctly converting ALL weights without loss.
"""

import torch
import json
from pathlib import Path
from safetensors.torch import save_file
from model import GPTConfig


def convert_checkpoint_to_safetensor(
    checkpoint_path,
    output_dir,
    model_name="gpt2-instruct",
    vocab_size=50304,
    dtype="bfloat16"
):
    """
    Convert PyTorch checkpoint to Hugging Face safetensors format.
    """
    print("="*80)
    print("PyTorch to Safetensor Converter (v2)")
    print("="*80)
    print(f"Input checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output dtype: {dtype}")
    print()
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print(f"✓ Found 'model' key in checkpoint")
    else:
        raise ValueError("Checkpoint must contain 'model' key with state dict")
    
    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    total_size = sum(p.numel() * p.element_size() for p in state_dict.values() if isinstance(p, torch.Tensor))
    print(f"  State dict keys: {len(state_dict)}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Estimated size (float32): {total_size / (1024**3):.2f} GB")
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        args = checkpoint.get('args', {})
        config = GPTConfig(
            vocab_size=vocab_size,
            context_length=args.get('context_length', 1024),
            num_layers=args.get('num_layers', 12),
            num_heads=args.get('num_heads', 12),
            embd_size=args.get('embd_size', 768)
        )
    
    print(f"✓ Model config: {config.num_layers} layers, {config.embd_size} dim, {config.num_heads} heads")
    print(f"  Vocab size: {config.vocab_size}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert state dict - transpose Conv1D weights for Hugging Face
    print("Converting state dict...")
    print("  - Transposing Conv1D weights (c_attn, c_proj, c_fc)")
    print("  - Handling shared weights (lm_head <-> wte)")
    if dtype != "float32":
        print(f"  - Converting to {dtype}")
    print()
    
    # Keys that need transposing
    transpose_suffixes = [
        '.attn.c_attn.weight',
        '.attn.c_proj.weight',
        '.mlp.c_fc.weight',
        '.mlp.c_proj.weight'
    ]
    
    hf_state_dict = {}
    transposed_count = 0
    non_tensor_count = 0
    
    for key, value in state_dict.items():
        # Skip non-tensors
        if not isinstance(value, torch.Tensor):
            non_tensor_count += 1
            if non_tensor_count <= 5:  # Show first few
                print(f"  ⚠ Skipping non-tensor: {key} (type: {type(value).__name__})")
            continue
        
        # Check if needs transposing
        needs_transpose = any(key.endswith(suffix) for suffix in transpose_suffixes)
        
        if needs_transpose:
            # Transpose for Hugging Face Conv1D format
            tensor = value.T.clone().contiguous()
            transposed_count += 1
            if transposed_count <= 4:
                print(f"    Transposed: {key} {value.shape} -> {tensor.shape}")
        elif key == 'lm_head.weight' or key == 'transformer.wte.weight':
            # Clone shared weights to make independent
            tensor = value.clone().contiguous()
        else:
            # Other weights stay the same (just make contiguous)
            tensor = value.contiguous()
        
        # Convert dtype
        if dtype == "float16":
            tensor = tensor.half()
        elif dtype == "bfloat16":
            tensor = tensor.bfloat16()
        elif dtype == "float32":
            tensor = tensor.float()
        
        hf_state_dict[key] = tensor
    
    if non_tensor_count > 5:
        print(f"  ... and {non_tensor_count - 5} more non-tensor items skipped")
    
    # Verify conversion
    total_params_after = sum(p.numel() for p in hf_state_dict.values())
    total_size_after = sum(p.numel() * p.element_size() for p in hf_state_dict.values())
    
    print(f"\n✓ Converted {len(hf_state_dict)} weight tensors")
    print(f"  Transposed {transposed_count} Conv1D weights")
    print(f"  Total parameters: {total_params_after:,}")
    print(f"  Estimated size: {total_size_after / (1024**3):.2f} GB")
    
    if total_params != total_params_after:
        print(f"  ⚠ WARNING: Parameter count changed!")
        print(f"     Before: {total_params:,}")
        print(f"     After: {total_params_after:,}")
        print(f"     Difference: {total_params - total_params_after:,}")
    print()
    
    # Save safetensors
    print("Saving safetensors...")
    safetensors_path = output_path / "model.safetensors"
    save_file(hf_state_dict, safetensors_path)
    file_size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    file_size_gb = safetensors_path.stat().st_size / (1024 ** 3)
    print(f"✓ Saved: {safetensors_path}")
    print(f"  File size: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")
    print()
    
    # Create config.json
    print("Creating config.json...")
    hf_config = {
        "architectures": ["GPT2LMHeadModel"],
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-5,
        "model_type": "gpt2",
        "n_ctx": config.context_length,
        "n_embd": config.embd_size,
        "n_head": config.num_heads,
        "n_inner": 4 * config.embd_size,
        "n_layer": config.num_layers,
        "n_positions": config.context_length,
        "reorder_and_upcast_attn": False,
        "scale_attn_weights": True,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "task_specific_params": {
            "text-generation": {
                "do_sample": True,
                "max_length": 50
            }
        },
        "torch_dtype": dtype,
        "transformers_version": "4.21.0",
        "use_cache": True,
        "vocab_size": config.vocab_size  # Keep original vocab size
    }
    
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(hf_config, f, indent=2)
    print(f"✓ Saved: {config_path}")
    print()
    
    # Create tokenizer config
    print("Creating tokenizer_config.json...")
    tokenizer_config = {
        "add_prefix_space": False,
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "model_max_length": config.context_length,
        "tokenizer_class": "GPT2Tokenizer",
        "unk_token": "<|endoftext|>"
    }
    
    tokenizer_config_path = output_path / "tokenizer_config.json"
    with open(tokenizer_config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"✓ Saved: {tokenizer_config_path}")
    print()
    
    print("="*80)
    print("✓ Conversion Complete!")
    print("="*80)
    print(f"Model saved to: {output_path}")
    print()
    print("Note: Model has vocab_size={config.vocab_size}, but GPT2Tokenizer uses vocab_size=50257.")
    print("      This is expected - the model will use the full vocabulary.")
    print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to Hugging Face safetensors")
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='hf_model', help='Output directory')
    parser.add_argument('--model_name', type=str, default='gpt2-instruct', help='Model name')
    parser.add_argument('--vocab_size', type=int, default=50304, help='Vocabulary size')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16', 'bfloat16'], 
                       default='bfloat16', help='Output dtype')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1
    
    convert_checkpoint_to_safetensor(
        checkpoint_path=str(checkpoint_path),
        output_dir=args.output_dir,
        model_name=args.model_name,
        vocab_size=args.vocab_size,
        dtype=args.dtype
    )
    
    return 0


if __name__ == "__main__":
    exit(main())

