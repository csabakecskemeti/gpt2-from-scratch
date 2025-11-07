#!/usr/bin/env python3
"""
Utility to generate safe resume commands from checkpoints.
Ensures you resume with the exact same hyperparameters to avoid accidents.

Usage:
    python generate_resume_command.py checkpoints/rolling_step_014000.pt
    python generate_resume_command.py checkpoints/rolling_step_014000.pt --output script
    python generate_resume_command.py checkpoints/rolling_step_014000.pt --compare --mini_batch_size 64
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Hyperparameters that should be preserved when resuming
RESUME_HYPERPARAMS = [
    'total_batch_size',
    'mini_batch_size',
    'context_length',
    'num_layers',
    'embd_size',
    'num_heads',
    'max_lr',
    'min_lr',
    'warmup_steps',
    'weight_decay',
    'num_epochs',
    'steps_per_epoch',
    'eval_freq',
    'checkpoint_freq',
    'keep_checkpoints',
    'seed',
]

# Optional parameters (don't need to be preserved)
OPTIONAL_PARAMS = [
    'use_tensorboard',
    'tensorboard_dir',
    'run_name',
    'logdir',
    'checkpoint_dir',
]


def validate_checkpoint(checkpoint_path):
    """Validate checkpoint and return checkpoint data"""
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        return None
    
    # Check file size (should be > 100MB for a real checkpoint)
    file_size = os.path.getsize(checkpoint_path)
    if file_size < 1_000_000:  # Less than 1MB is suspicious
        print(f"⚠️  Warning: Checkpoint file is very small ({file_size:,} bytes)")
        print(f"   This might be a corrupted checkpoint!")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return None
    
    # Try to load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None
    
    # Validate required keys
    required_keys = ['step', 'epoch', 'model', 'optimizer', 'args']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    
    if missing_keys:
        print(f"❌ Error: Checkpoint is missing required keys: {missing_keys}")
        return None
    
    print(f"✅ Checkpoint is valid!")
    return checkpoint


def print_checkpoint_info(checkpoint, checkpoint_path):
    """Print detailed checkpoint information"""
    
    print(f"\n{'='*70}")
    print(f"📋 CHECKPOINT INFORMATION")
    print(f"{'='*70}")
    print(f"File: {checkpoint_path}")
    print(f"Size: {os.path.getsize(checkpoint_path) / 1024**2:.1f} MB")
    print(f"\n📊 Training Progress:")
    print(f"   Step:        {checkpoint['step']:,}")
    print(f"   Epoch:       {checkpoint['epoch']}")
    print(f"   Train Loss:  {checkpoint.get('train_loss', 'N/A'):.4f}" if isinstance(checkpoint.get('train_loss'), (int, float)) else f"   Train Loss:  N/A")
    print(f"   Val Loss:    {checkpoint.get('val_loss', 'N/A'):.4f}" if isinstance(checkpoint.get('val_loss'), (int, float)) else f"   Val Loss:    N/A")
    
    if 'timestamp' in checkpoint:
        timestamp = datetime.fromtimestamp(checkpoint['timestamp'])
        print(f"   Saved:       {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n⚙️  Hyperparameters:")
    args = checkpoint['args']
    for param in RESUME_HYPERPARAMS:
        if param in args:
            value = args[param]
            print(f"   {param:20s} = {value}")
    
    print(f"{'='*70}\n")


def generate_resume_command(checkpoint, checkpoint_path, num_gpus=1, override_params=None):
    """Generate the exact resume command"""
    
    args = checkpoint['args']
    
    # Start with base command
    if num_gpus > 1:
        cmd_parts = [
            f"torchrun --standalone --nproc_per_node={num_gpus} src/train_improved.py"
        ]
    else:
        cmd_parts = [
            "python src/train_improved.py"
        ]
    
    # Add resume flag
    cmd_parts.append(f"--resume {checkpoint_path}")
    
    # Add all hyperparameters from checkpoint
    for param in RESUME_HYPERPARAMS:
        if param in args:
            value = args[param]
            
            # Check if this parameter is being overridden
            if override_params and param in override_params:
                value = override_params[param]
            
            cmd_parts.append(f"--{param} {value}")
    
    # Add optional parameters if present
    for param in OPTIONAL_PARAMS:
        if param in args and args[param]:
            value = args[param]
            
            # Check if this parameter is being overridden
            if override_params and param in override_params:
                value = override_params[param]
            
            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{param}")
            else:
                cmd_parts.append(f"--{param} {value}")
    
    return " \\\n    ".join(cmd_parts)


def compare_params(checkpoint, override_params):
    """Compare checkpoint params with override params and show differences"""
    
    if not override_params:
        return
    
    args = checkpoint['args']
    differences = []
    
    for param, new_value in override_params.items():
        if param in args:
            old_value = args[param]
            if old_value != new_value:
                differences.append((param, old_value, new_value))
    
    if differences:
        print(f"\n{'='*70}")
        print(f"⚠️  WARNING: PARAMETER CHANGES DETECTED")
        print(f"{'='*70}")
        print(f"{'Parameter':<25} {'Checkpoint':<20} {'Override':<20}")
        print(f"{'-'*70}")
        for param, old_val, new_val in differences:
            print(f"{param:<25} {str(old_val):<20} {str(new_val):<20}")
        print(f"{'='*70}")
        print(f"\n⚠️  Changing hyperparameters during resume can affect training!")
        print(f"   - Optimizer state was built with old parameters")
        print(f"   - Learning rate schedule may be affected")
        print(f"   - Gradient accumulation steps may change")
        print(f"\nRecommendation: Only change parameters if you know what you're doing.")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate safe resume commands from checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - generate resume command
  python generate_resume_command.py checkpoints/rolling_step_014000.pt
  
  # Generate command for 4-GPU training
  python generate_resume_command.py checkpoints/rolling_step_014000.pt --num_gpus 4
  
  # Save command to shell script
  python generate_resume_command.py checkpoints/rolling_step_014000.pt --output script
  
  # Compare with different parameters
  python generate_resume_command.py checkpoints/rolling_step_014000.pt --compare --mini_batch_size 64
  
  # Generate command with overrides (USE WITH CAUTION!)
  python generate_resume_command.py checkpoints/rolling_step_014000.pt --mini_batch_size 64 --output command
        """
    )
    
    parser.add_argument("checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of GPUs to use (1 for single GPU, >1 for multi-GPU)")
    parser.add_argument("--output", choices=['command', 'script', 'info'], default='command',
                       help="output format: 'command' (just command), 'script' (save as .sh), 'info' (show info only)")
    parser.add_argument("--compare", action='store_true', help="compare checkpoint params with any overrides provided")
    parser.add_argument("--script_name", type=str, default="resume_training_safe.sh", help="name for output script file")
    
    # Allow overriding any hyperparameter (with warning)
    parser.add_argument("--total_batch_size", type=int, help="override total_batch_size")
    parser.add_argument("--mini_batch_size", type=int, help="override mini_batch_size")
    parser.add_argument("--context_length", type=int, help="override context_length")
    parser.add_argument("--num_layers", type=int, help="override num_layers")
    parser.add_argument("--embd_size", type=int, help="override embd_size")
    parser.add_argument("--num_heads", type=int, help="override num_heads")
    parser.add_argument("--max_lr", type=float, help="override max_lr")
    parser.add_argument("--min_lr", type=float, help="override min_lr")
    parser.add_argument("--warmup_steps", type=int, help="override warmup_steps")
    parser.add_argument("--weight_decay", type=float, help="override weight_decay")
    parser.add_argument("--num_epochs", type=int, help="override num_epochs")
    parser.add_argument("--steps_per_epoch", type=int, help="override steps_per_epoch")
    parser.add_argument("--eval_freq", type=int, help="override eval_freq")
    parser.add_argument("--checkpoint_freq", type=int, help="override checkpoint_freq")
    parser.add_argument("--keep_checkpoints", type=int, help="override keep_checkpoints")
    parser.add_argument("--seed", type=int, help="override seed")
    parser.add_argument("--use_tensorboard", action='store_true', help="enable TensorBoard")
    parser.add_argument("--tensorboard_dir", type=str, help="override tensorboard_dir")
    parser.add_argument("--run_name", type=str, help="override run_name")
    
    args = parser.parse_args()
    
    # Validate checkpoint
    checkpoint = validate_checkpoint(args.checkpoint)
    if checkpoint is None:
        sys.exit(1)
    
    # Collect any override parameters
    override_params = {}
    for param in RESUME_HYPERPARAMS + OPTIONAL_PARAMS:
        value = getattr(args, param, None)
        if value is not None:
            override_params[param] = value
    
    # Show checkpoint info unless output is 'command' only
    if args.output != 'command' or args.compare:
        print_checkpoint_info(checkpoint, args.checkpoint)
    
    # Compare parameters if requested
    if args.compare or override_params:
        compare_params(checkpoint, override_params)
    
    # Generate command
    resume_command = generate_resume_command(
        checkpoint, 
        args.checkpoint, 
        num_gpus=args.num_gpus,
        override_params=override_params if override_params else None
    )
    
    # Output based on format
    if args.output == 'info':
        # Info only, no command
        pass
    
    elif args.output == 'command':
        # Print command to stdout
        print(f"\n{'='*70}")
        print(f"🚀 RESUME COMMAND")
        print(f"{'='*70}")
        print(resume_command)
        print(f"{'='*70}\n")
        
        print(f"💡 To run: Copy and paste the command above")
        print(f"💡 To save: Use --output script")
    
    elif args.output == 'script':
        # Save to shell script
        script_content = f"""#!/bin/bash
# Auto-generated resume script
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Checkpoint: {args.checkpoint}
# Step: {checkpoint['step']:,}
# Epoch: {checkpoint['epoch']}

cd /home/kecso/Documents/workspace/training/gpt-2/gpt2-from-scratch

{resume_command}
"""
        
        script_path = Path(args.script_name)
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"\n{'='*70}")
        print(f"✅ SCRIPT SAVED")
        print(f"{'='*70}")
        print(f"File: {script_path.absolute()}")
        print(f"Size: {os.path.getsize(script_path)} bytes")
        print(f"\n🚀 To run:")
        print(f"   ./{args.script_name}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

