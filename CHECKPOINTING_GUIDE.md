# Checkpointing System Guide

## Overview

The improved training script (`train_improved.py`) implements a **smart 3-tier checkpointing system** designed for long-running training (like your 34-day run!).

## ✅ Key Improvements Over Original

| Feature | Original `train.py` | Improved `train_improved.py` |
|---------|-------------------|---------------------------|
| **Save Frequency** | Every 10,000 steps | Every 250 steps (configurable) |
| **Resume Training** | ❌ Not supported | ✅ Full resume support |
| **Optimizer State** | ❌ Not saved | ✅ Saved |
| **RNG States** | ❌ Not saved | ✅ Saved (reproducible) |
| **DataLoader Position** | ❌ Not saved | ✅ Saved (no data repeat) |
| **Checkpoint Cleanup** | ❌ None (fills disk) | ✅ Automatic cleanup |
| **Best Model Tracking** | ❌ No | ✅ Automatically saved |
| **Graceful Shutdown** | ❌ No | ✅ Saves on Ctrl+C |
| **Epoch Checkpoints** | ❌ No | ✅ One per epoch |

---

## 📁 Checkpoint Directory Structure

After running the improved training, your checkpoint directory will look like:

```
checkpoints/
├── latest.pt                    # Most recent checkpoint (always overwritten)
├── latest_backup.pt             # Backup of previous latest (safety net)
├── best_model.pt                # Best validation loss checkpoint
├── epoch_00001.pt               # End of epoch 1
├── epoch_00002.pt               # End of epoch 2
├── epoch_00003.pt               # End of epoch 3
├── epoch_00004.pt               # End of epoch 4
├── epoch_00005.pt               # End of epoch 5
├── rolling_step_094500.pt       # Rolling checkpoint
├── rolling_step_094750.pt       # (keeps last 10 only)
├── rolling_step_095000.pt
└── ...
```

**Disk Usage Estimate:**
- Each checkpoint: ~500 MB (for GPT-2 124M)
- 10 rolling + 5 epoch + 2 latest + 1 best = **~9 GB total**

---

## 🚀 Usage

### Starting Fresh Training

```bash
# Single GPU
python src/train_improved.py

# Multi-GPU (4 GPUs)
torchrun --standalone --nproc_per_node=4 src/train_improved.py
```

### Resuming Training

**From latest checkpoint (most common):**
```bash
python src/train_improved.py --resume latest
```

**From best model:**
```bash
python src/train_improved.py --resume best
```

**From specific checkpoint:**
```bash
python src/train_improved.py --resume checkpoints/epoch_00003.pt
```

**Multi-GPU resume:**
```bash
torchrun --standalone --nproc_per_node=4 src/train_improved.py --resume latest
```

### Listing Available Checkpoints

```bash
python src/train_improved.py --list_checkpoints
```

**Output example:**
```
Available checkpoints:
------------------------------------------------------------
  latest: step=5000, val_loss=3.2345
  best:   step=4750, val_loss=3.1234

Epoch checkpoints:
  epoch_00001.pt: step=19073, val_loss=3.4567
  epoch_00002.pt: step=38146, val_loss=3.2890

Rolling checkpoints (last 10):
  rolling_step_004250.pt: step=4250, val_loss=3.2456
  rolling_step_004500.pt: step=4500, val_loss=3.2234
  rolling_step_004750.pt: step=4750, val_loss=3.1234
  rolling_step_005000.pt: step=5000, val_loss=3.2345
------------------------------------------------------------
```

---

## ⚙️ Configuration Options

### Checkpoint Frequency

Control how often checkpoints are saved:

```bash
# Save every 100 steps (more frequent, safer)
python src/train_improved.py --checkpoint_freq 100

# Save every 500 steps (less frequent, fewer files)
python src/train_improved.py --checkpoint_freq 500
```

**Recommendation for 34-day training:** 250-500 steps

### Number of Rolling Checkpoints to Keep

```bash
# Keep last 20 checkpoints (uses more disk)
python src/train_improved.py --keep_checkpoints 20

# Keep last 5 checkpoints (uses less disk)
python src/train_improved.py --keep_checkpoints 5
```

**Recommendation:** 10 (default) provides good balance

### Checkpoint Directory

```bash
python src/train_improved.py --checkpoint_dir /path/to/checkpoints/
```

---

## 🛡️ What's Saved in Each Checkpoint?

Each checkpoint contains **complete training state**:

```python
checkpoint = {
    'step': 5000,                           # Current training step
    'epoch': 1,                             # Current epoch
    'model': model.state_dict(),            # Model weights
    'optimizer': optimizer.state_dict(),    # Optimizer state (momentum, etc.)
    'train_loss': 3.2345,                   # Training loss
    'val_loss': 3.1234,                     # Validation loss
    'rng_state': {                          # RNG states for reproducibility
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
    },
    'dataloader_state': {                   # DataLoader position
        'train_curr_shard': 2,
        'train_curr_pos': 12345678,
        'val_curr_shard': 0,
        'val_curr_pos': 5432,
    },
    'args': {...},                          # All hyperparameters
    'timestamp': 1699123456.789,            # When checkpoint was saved
}
```

This means when you resume, **everything** is exactly as it was - including:
- ✅ Model weights
- ✅ Optimizer momentum buffers
- ✅ Random number generators (reproducibility)
- ✅ DataLoader position (no duplicate or skipped data)
- ✅ Learning rate schedule (continues from correct point)

---

## 🎯 Checkpoint Types Explained

### 1. Latest Checkpoint (`latest.pt`)

**Purpose:** Quick resume from most recent state

**Behavior:**
- Overwrites on every save
- Always represents the most recent training state
- Has a backup (`latest_backup.pt`) in case of corruption

**Use case:** Resume after crash, power loss, or manual stop

### 2. Rolling Checkpoints (`rolling_step_XXXXXX.pt`)

**Purpose:** Keep recent history for recovery

**Behavior:**
- Saved every `checkpoint_freq` steps
- Keeps only last N (default: 10)
- Automatically deletes older ones

**Use case:**
- Resume from slightly earlier point if latest is corrupted
- Compare model performance at different recent steps

**Example timeline:**
```
Step 10000: Create rolling_step_010000.pt
Step 10250: Create rolling_step_010250.pt
...
Step 12500: Create rolling_step_012500.pt
           Delete rolling_step_010000.pt (oldest, beyond keep_last_n)
```

### 3. Epoch Checkpoints (`epoch_XXXXX.pt`)

**Purpose:** Major milestones, keep forever

**Behavior:**
- Saved at end of each epoch
- Never deleted automatically
- Represents complete pass through dataset

**Use case:**
- Compare model across epochs
- Resume from epoch boundaries
- Long-term model archiving

### 4. Best Model Checkpoint (`best_model.pt`)

**Purpose:** Save the best performing model

**Behavior:**
- Overwrites when validation loss improves
- Always represents the best model so far

**Use case:**
- Deploy the best model
- Resume from best point if training diverges later

---

## 🚨 Graceful Shutdown (Ctrl+C Handling)

The improved trainer handles interrupts gracefully:

```
^C
============================================================
⚠️  Interrupt signal received! Saving checkpoint before exit...
============================================================

Saving emergency checkpoint...
✓ Saved latest checkpoint: checkpoints/latest.pt
✓ Emergency checkpoint saved. Exiting...
```

**What happens:**
1. Catches Ctrl+C signal
2. Saves current state as `latest.pt`
3. Cleanly exits (no corruption)

**Resume after interrupt:**
```bash
python src/train_improved.py --resume latest
```

---

## 🔧 Troubleshooting

### Problem: "Checkpoint not found"

**Check available checkpoints:**
```bash
python src/train_improved.py --list_checkpoints
```

**Common causes:**
- Wrong checkpoint directory
- Typo in checkpoint path
- No checkpoints saved yet (if training just started)

### Problem: "CUDA out of memory" when resuming

**Cause:** Checkpoint saved with different GPU configuration

**Solution:** Resume with same number of GPUs as original training:
```bash
# If original training used 4 GPUs
torchrun --standalone --nproc_per_node=4 src/train_improved.py --resume latest
```

### Problem: Disk filling up

**Check disk usage:**
```bash
du -sh checkpoints/
```

**Solutions:**
1. Reduce `--keep_checkpoints` (default: 10)
2. Manually delete old epoch checkpoints you don't need
3. Move old checkpoints to slower/cheaper storage

### Problem: Want to change hyperparameters when resuming

**You can!** The script will use new CLI arguments for most settings:

```bash
# Resume but change learning rate
python src/train_improved.py --resume latest --max_lr 5e-4

# Resume but change evaluation frequency  
python src/train_improved.py --resume latest --eval_freq 100
```

**Note:** Some settings can't be changed (e.g., model architecture, batch size) because they're baked into the saved state.

---

## 📊 Monitoring Checkpoint Health

### Check a Checkpoint's Contents

```python
import torch

# Load checkpoint
ckpt = torch.load('checkpoints/latest.pt', map_location='cpu')

# Inspect
print(f"Step: {ckpt['step']}")
print(f"Epoch: {ckpt['epoch']}")
print(f"Val Loss: {ckpt['val_loss']:.4f}")
print(f"Train Loss: {ckpt['train_loss']:.4f}")
print(f"Timestamp: {ckpt['timestamp']}")
print(f"DataLoader: shard={ckpt['dataloader_state']['train_curr_shard']}, pos={ckpt['dataloader_state']['train_curr_pos']}")
```

### Verify Checkpoint Integrity

```python
import torch

try:
    ckpt = torch.load('checkpoints/latest.pt', map_location='cpu')
    print("✓ Checkpoint is valid")
    print(f"  Step: {ckpt['step']}")
    print(f"  Val Loss: {ckpt['val_loss']:.4f}")
except Exception as e:
    print(f"✗ Checkpoint is corrupted: {e}")
```

---

## 💡 Best Practices for 34-Day Training

### Recommended Settings

```bash
torchrun --standalone --nproc_per_node=4 src/train_improved.py \
  --checkpoint_freq 250 \          # Save every 250 steps (~1 minute)
  --keep_checkpoints 10 \           # Keep last 10 rolling checkpoints
  --eval_freq 250 \                 # Evaluate every 250 steps
  --checkpoint_dir /fast/disk/checkpoints/  # Use fast SSD
```

### Periodic Manual Backups

Even with automatic checkpointing, consider manual backups:

```bash
# Every few days, backup to safe storage
cp checkpoints/latest.pt /backup/location/checkpoint_day3.pt
cp checkpoints/best_model.pt /backup/location/best_model_day3.pt
```

### Monitor Disk Space

```bash
# Check disk space regularly
df -h /path/to/checkpoints

# Set up alert if disk gets too full
```

### Test Resume Early

Don't wait until you need it! Test resume functionality early:

```bash
# Start training
python src/train_improved.py

# After 10 minutes, press Ctrl+C
^C

# Resume immediately
python src/train_improved.py --resume latest

# Verify training continues smoothly
```

---

## 📈 Performance Impact

**Checkpoint overhead:**
- Saving one checkpoint: ~1-2 seconds (500 MB to SSD)
- Frequency: Every 250 steps = ~50 seconds of training
- **Impact: <5% overhead** (negligible)

**Loading checkpoint:**
- Loading on startup: ~2-3 seconds
- **One-time cost**, doesn't affect training speed

---

## 🔄 Migrating from Original `train.py`

If you already started training with original `train.py`:

**Option 1: Start Fresh (Recommended)**
```bash
# The old checkpoints are incomplete (no optimizer state)
# Better to start from scratch with improved script
python src/train_improved.py
```

**Option 2: Continue from Old Checkpoint (Advanced)**

You'd need to manually convert old checkpoint format. Not recommended unless you've trained for a very long time already.

---

## 📚 Summary

| Scenario | Command |
|----------|---------|
| Start fresh training | `python src/train_improved.py` |
| Resume from latest | `python src/train_improved.py --resume latest` |
| Resume from best | `python src/train_improved.py --resume best` |
| List checkpoints | `python src/train_improved.py --list_checkpoints` |
| Multi-GPU training | `torchrun --nproc_per_node=4 src/train_improved.py` |
| Multi-GPU resume | `torchrun --nproc_per_node=4 src/train_improved.py --resume latest` |

---

## 🎉 You're All Set!

The improved checkpointing system gives you:
- ✅ **Safety**: Never lose more than a few minutes of training
- ✅ **Flexibility**: Resume from any saved point
- ✅ **Reproducibility**: Exact state restoration including RNG
- ✅ **Convenience**: Automatic cleanup, best model tracking
- ✅ **Peace of mind**: Graceful shutdown on Ctrl+C

Your 34-day training is now much safer! 🚀

