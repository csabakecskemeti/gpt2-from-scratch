# Resume Command Generator 🚀

A safety utility that generates exact resume commands from checkpoints, ensuring you don't accidentally change hyperparameters.

## Why This Tool?

**Problem:** When resuming training, it's easy to accidentally change hyperparameters:
```bash
# Original training: mini_batch_size=32
python src/train_improved.py --mini_batch_size 32 ...

# Oops! Forgot what the original params were!
python src/train_improved.py --resume latest --mini_batch_size 64  # CHANGED!
```

**Solution:** This tool reads the checkpoint and generates the EXACT command to resume safely.

---

## Quick Start

### 1. Basic Usage - Generate Resume Command
```bash
cd /home/kecso/Documents/workspace/training/gpt-2/gpt2-from-scratch
source .venv/bin/activate

python generate_resume_command.py checkpoints/rolling_step_014000.pt
```

**Output:**
```bash
✅ Checkpoint is valid!

🚀 RESUME COMMAND
======================================================================
python src/train_improved.py \
    --resume checkpoints/rolling_step_014000.pt \
    --total_batch_size 524288 \
    --mini_batch_size 32 \
    --context_length 1024 \
    --num_layers 12 \
    --embd_size 768 \
    --num_heads 12 \
    --max_lr 0.001 \
    --min_lr 0.0001 \
    --warmup_steps 715 \
    --weight_decay 0.1 \
    --num_epochs 5 \
    --steps_per_epoch 19073 \
    --eval_freq 250 \
    --checkpoint_freq 250 \
    --keep_checkpoints 10 \
    --seed 1337 \
    --use_tensorboard \
    --tensorboard_dir ./runs/ \
    --logdir ./logs/
======================================================================
```

Just copy and paste! 🎯

---

## Usage Examples

### Example 1: Multi-GPU Training (4 GPUs)
```bash
python generate_resume_command.py checkpoints/rolling_step_014000.pt --num_gpus 4
```

Generates:
```bash
torchrun --standalone --nproc_per_node=4 src/train_improved.py \
    --resume checkpoints/rolling_step_014000.pt \
    [... all original hyperparameters ...]
```

### Example 2: Save to Shell Script
```bash
python generate_resume_command.py checkpoints/rolling_step_014000.pt \
    --output script \
    --script_name resume_from_14000.sh
```

Then run:
```bash
./resume_from_14000.sh
```

### Example 3: Check Checkpoint Info Only
```bash
python generate_resume_command.py checkpoints/rolling_step_014000.pt --output info
```

Shows:
```
📋 CHECKPOINT INFORMATION
======================================================================
File: checkpoints/rolling_step_014000.pt
Size: 456.7 MB

📊 Training Progress:
   Step:        14,000
   Epoch:       0
   Train Loss:  3.4567
   Val Loss:    3.2345
   Saved:       2025-11-06 14:30:22

⚙️  Hyperparameters:
   total_batch_size     = 524288
   mini_batch_size      = 32
   context_length       = 1024
   num_layers           = 12
   embd_size            = 768
   [...]
======================================================================
```

### Example 4: Compare Before Changing Parameters (SAFETY CHECK!)
```bash
python generate_resume_command.py checkpoints/rolling_step_014000.pt \
    --compare \
    --mini_batch_size 64
```

Shows:
```
⚠️  WARNING: PARAMETER CHANGES DETECTED
======================================================================
Parameter                 Checkpoint           Override            
----------------------------------------------------------------------
mini_batch_size           32                   64                  
======================================================================

⚠️  Changing hyperparameters during resume can affect training!
   - Optimizer state was built with old parameters
   - Learning rate schedule may be affected
   - Gradient accumulation steps may change

Recommendation: Only change parameters if you know what you're doing.
======================================================================
```

Then generates the command with the override (if you're sure).

---

## Command-Line Options

### Basic Options
```bash
python generate_resume_command.py <checkpoint_path> [options]
```

| Option | Description |
|--------|-------------|
| `checkpoint_path` | Path to checkpoint (required) |
| `--num_gpus N` | Number of GPUs (1=single GPU, >1=multi-GPU) |
| `--output {command,script,info}` | Output format (default: command) |
| `--script_name FILE` | Name for output script (default: resume_training_safe.sh) |
| `--compare` | Show parameter differences if overriding |

### Override Options (USE WITH CAUTION!)
```bash
--total_batch_size N
--mini_batch_size N
--context_length N
--num_layers N
--embd_size N
--num_heads N
--max_lr FLOAT
--min_lr FLOAT
--warmup_steps N
--weight_decay FLOAT
--num_epochs N
--steps_per_epoch N
--eval_freq N
--checkpoint_freq N
--keep_checkpoints N
--seed N
--use_tensorboard
--tensorboard_dir DIR
--run_name NAME
```

**⚠️ Warning:** Overriding hyperparameters can affect training quality! The tool will warn you about any changes.

---

## Safety Features

### ✅ Checkpoint Validation
- Checks if file exists
- Verifies file size (warns if < 1MB, likely corrupted)
- Attempts to load checkpoint
- Validates required keys (step, epoch, model, optimizer, args)

### ⚠️ Parameter Change Detection
- Compares checkpoint params with any overrides
- Shows clear diff table
- Warns about potential issues
- Recommends safe usage

### 🛡️ Safe by Default
- Default: Generates command with EXACT original parameters
- Override: Only if you explicitly provide different values
- Compare: Shows you exactly what would change

---

## Common Workflows

### Workflow 1: Safe Resume (Recommended)
```bash
# 1. Generate safe command
python generate_resume_command.py checkpoints/latest.pt

# 2. Copy the generated command and run it
python src/train_improved.py --resume checkpoints/latest.pt [...]
```

### Workflow 2: Resume with Script
```bash
# 1. Generate script
python generate_resume_command.py checkpoints/rolling_step_014000.pt \
    --output script \
    --num_gpus 4

# 2. Run script
./resume_training_safe.sh
```

### Workflow 3: Intentional Parameter Change
```bash
# 1. Compare first to see what changes
python generate_resume_command.py checkpoints/rolling_step_014000.pt \
    --compare \
    --mini_batch_size 64

# 2. Review the warnings

# 3. If you're sure, generate the command
python generate_resume_command.py checkpoints/rolling_step_014000.pt \
    --mini_batch_size 64 \
    --output script

# 4. Run it
./resume_training_safe.sh
```

### Workflow 4: Check Multiple Checkpoints
```bash
# Compare different checkpoints
python generate_resume_command.py checkpoints/rolling_step_014000.pt --output info
python generate_resume_command.py checkpoints/rolling_step_014500.pt --output info
python generate_resume_command.py checkpoints/best_model.pt --output info

# Choose the best one and generate command
python generate_resume_command.py checkpoints/best_model.pt
```

---

## Examples from Your Training

### Resume from Latest (Single GPU)
```bash
python generate_resume_command.py checkpoints/latest.pt
```

### Resume from Latest (4 GPUs)
```bash
python generate_resume_command.py checkpoints/latest.pt --num_gpus 4
```

### Resume from Specific Rolling Checkpoint
```bash
python generate_resume_command.py checkpoints/rolling_step_014000.pt
```

### Resume from Best Model
```bash
python generate_resume_command.py checkpoints/best_model.pt
```

### Resume with Increased Batch Size (After GPU Upgrade)
```bash
# First, check what changes
python generate_resume_command.py checkpoints/latest.pt \
    --compare \
    --mini_batch_size 64

# Then generate if safe
python generate_resume_command.py checkpoints/latest.pt \
    --mini_batch_size 64 \
    --output script
```

---

## Troubleshooting

### Error: "Checkpoint not found"
```bash
❌ Error: Checkpoint not found: checkpoints/rolling_step_014000.pt
```
**Solution:** Check the path. Use `ls checkpoints/` to see available checkpoints.

### Warning: "Checkpoint file is very small"
```bash
⚠️  Warning: Checkpoint file is very small (1,234 bytes)
   This might be a corrupted checkpoint!
   Continue anyway? (y/N):
```
**Solution:** This checkpoint is likely corrupted. Choose a different one.

### Error: "Checkpoint is missing required keys"
```bash
❌ Error: Checkpoint is missing required keys: ['args']
```
**Solution:** This checkpoint is incomplete or from an old version. Use a different checkpoint.

---

## Integration with Existing Tools

### With List Checkpoints
```bash
# 1. List available checkpoints
python src/train_improved.py --list_checkpoints

# 2. Choose one and generate resume command
python generate_resume_command.py checkpoints/rolling_step_014000.pt
```

### With TensorBoard
```bash
# Generate command with TensorBoard enabled
python generate_resume_command.py checkpoints/latest.pt --use_tensorboard
```

### With Checkpoint Manager
The generated commands work seamlessly with the existing checkpoint system.

---

## Tips & Best Practices

1. **Always use this tool when resuming** - Prevents accidental parameter changes
2. **Use `--compare` before overriding** - See exactly what will change
3. **Save as script for repeated use** - Easier than retyping
4. **Check checkpoint info first** - Verify step/epoch/loss before resuming
5. **Keep generated scripts** - Document your training runs

---

## Future Enhancements (Optional)

- [ ] Copy command to clipboard automatically
- [ ] Interactive mode (ask which parameters to override)
- [ ] Diff between two checkpoints
- [ ] Estimate training time from checkpoint
- [ ] Integration with git (save commit hash in checkpoint)

---

## See Also

- `CHECKPOINTING_GUIDE.md` - Detailed checkpointing documentation
- `TENSORBOARD_GUIDE.md` - TensorBoard monitoring guide
- `src/train_improved.py` - Main training script
- `list_checkpoints.sh` - List available checkpoints
- `resume_training.sh` - Original resume script (less safe)

---

**Created:** 2025-11-06  
**Author:** Auto-generated safety tool  
**Purpose:** Prevent accidental hyperparameter changes when resuming training

