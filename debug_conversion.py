#!/usr/bin/env python3
"""Debug script to check what's happening with the conversion."""

import torch
from safetensors.torch import load_file
from pathlib import Path

print("="*80)
print("Debugging Conversion")
print("="*80)

# Check checkpoint
print("\n1. Checking checkpoint...")
ckpt = torch.load('checkpoints_instruct/latest.pt', map_location='cpu', weights_only=False)
sd_ckpt = ckpt['model']
print(f"  Checkpoint state dict keys: {len(sd_ckpt)}")
print(f"  Checkpoint parameters: {sum(p.numel() for p in sd_ckpt.values() if isinstance(p, torch.Tensor)):,}")

# Check vocab-related weights
vocab_keys = [k for k in sd_ckpt.keys() if 'wte' in k or 'lm_head' in k]
print(f"\n  Vocab-related keys:")
for k in vocab_keys:
    print(f"    {k}: {sd_ckpt[k].shape}")

# Check safetensors
print("\n2. Checking safetensors...")
sd_safe = load_file('hf_model/model.safetensors')
print(f"  Safetensors keys: {len(sd_safe)}")
print(f"  Safetensors parameters: {sum(p.numel() for p in sd_safe.values()):,}")

# Compare keys
print("\n3. Comparing keys...")
ckpt_keys = set(sd_ckpt.keys())
safe_keys = set(sd_safe.keys())

missing_in_safe = ckpt_keys - safe_keys
extra_in_safe = safe_keys - ckpt_keys

if missing_in_safe:
    print(f"  ⚠ Keys in checkpoint but NOT in safetensors ({len(missing_in_safe)}):")
    for k in sorted(missing_in_safe)[:10]:
        print(f"    - {k}")
    if len(missing_in_safe) > 10:
        print(f"    ... and {len(missing_in_safe) - 10} more")

if extra_in_safe:
    print(f"  ⚠ Keys in safetensors but NOT in checkpoint ({len(extra_in_safe)}):")
    for k in sorted(extra_in_safe)[:10]:
        print(f"    + {k}")
    if len(extra_in_safe) > 10:
        print(f"    ... and {len(extra_in_safe) - 10} more")

# Check specific weight shapes
print("\n4. Checking weight shapes...")
key_samples = [
    'transformer.wte.weight',
    'lm_head.weight',
    'transformer.h.0.attn.c_attn.weight',
    'transformer.h.0.mlp.c_fc.weight',
]

for key in key_samples:
    if key in sd_ckpt and key in sd_safe:
        ckpt_shape = sd_ckpt[key].shape
        safe_shape = sd_safe[key].shape
        match = "✓" if ckpt_shape == safe_shape else "✗"
        print(f"  {match} {key}:")
        print(f"    Checkpoint: {ckpt_shape}")
        print(f"    Safetensors: {safe_shape}")

print("\n" + "="*80)

