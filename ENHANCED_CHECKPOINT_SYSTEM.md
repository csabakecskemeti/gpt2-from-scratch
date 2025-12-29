# Enhanced Checkpoint System 🛡️

**Version:** 2.0  
**Date:** 2025-11-06  
**Status:** ✅ Implemented

## Overview

The enhanced checkpoint system provides **enterprise-grade reliability** with:
- ✅ Atomic saves (corruption-proof)
- ✅ Automatic validation
- ✅ Separate rolling and emergency checkpoints
- ✅ Triple redundancy for best models
- ✅ Timestamped emergency saves
- ✅ Disk space monitoring

---

## Checkpoint Structure

### Complete Checkpoint Directory Layout

```
checkpoints/
├── latest.pt                              # Most recent checkpoint
├── latest_backup.pt                       # Backup of latest (corruption protection)
│
├── rolling_step_014000.pt                 # ┐
├── rolling_step_014500.pt                 #  │
├── rolling_step_015000.pt                 #  │ 10 rolling checkpoints
├── rolling_step_015500.pt                 #  │ (cleaned up automatically)
├── ...                                    #  │
├── rolling_step_019000.pt                 # ┘
├── rolling_last_backup.pt                 # Backup of most recent rolling
│
├── emergency_step_012345_20251106_143022.pt  # ┐ Emergency checkpoints
├── emergency_step_013456_20251106_150831.pt  #  │ (timestamped, separate cleanup)
├── emergency_step_014567_20251106_161245.pt  #  │ Saved on Ctrl+C
├── ...                                       #  │ (up to 10, separate from rolling)
├── emergency_step_019876_20251106_203145.pt  # ┘
│
├── best_model.pt                          # Current best model (lowest val loss)
├── best_model_backup.pt                   # Backup of current best (corruption protection)
├── best_model_previous.pt                 # Previous best (before it was replaced)
│
├── epoch_00000.pt                         # ┐ Epoch checkpoints
├── epoch_00001.pt                         #  │ (kept forever)
├── epoch_00002.pt                         #  │
├── ...                                    # ┘
```

---

## Checkpoint Types Explained

### 1. 📌 Latest Checkpoints

**Purpose:** Quick resume from the most recent training state.

| File | Description | Updated |
|------|-------------|---------|
| `latest.pt` | Current latest checkpoint | Every checkpoint save |
| `latest_backup.pt` | Previous latest (before overwrite) | Every checkpoint save |

**Recovery Scenario:**
```bash
# If latest.pt is corrupted
python src/train_improved.py --resume checkpoints/latest_backup.pt
```

**Use Case:** Normal resume operations.

---

### 2. 🔄 Rolling Checkpoints

**Purpose:** Periodic snapshots during training, with automatic cleanup.

| Feature | Details |
|---------|---------|
| **Naming** | `rolling_step_NNNNNN.pt` (6-digit step number) |
| **Frequency** | Every `--checkpoint_freq` steps (default: 250) |
| **Retention** | Keep last N (default: 10, configurable via `--keep_checkpoints`) |
| **Cleanup** | Automatic, oldest deleted when exceeding limit |
| **Backup** | `rolling_last_backup.pt` always contains copy of newest |

**Why Separate from Emergency?**
- Emergency saves (Ctrl+C) won't wipe out your regular rolling checkpoints
- You can stop training 10+ times without losing periodic snapshots

**Recovery Scenario:**
```bash
# Resume from specific rolling checkpoint
python src/train_improved.py --resume checkpoints/rolling_step_015000.pt

# Or from rolling backup if latest rolling is corrupted
python src/train_improved.py --resume checkpoints/rolling_last_backup.pt
```

**Use Case:** 
- Go back to a specific training step
- Recover if latest checkpoint is corrupted
- Analysis of training progression

---

### 3. 🚨 Emergency Checkpoints

**Purpose:** Save on unexpected interruptions (Ctrl+C, system shutdown).

| Feature | Details |
|---------|---------|
| **Naming** | `emergency_step_NNNNNN_YYYYMMDD_HHMMSS.pt` |
| **Frequency** | On interruption (Ctrl+C, SIGINT, SIGTERM) |
| **Retention** | Keep last N (default: 10, configurable via `--keep_emergency_checkpoints`) |
| **Cleanup** | Automatic, separate from rolling checkpoints |
| **Timestamp** | Includes save time for debugging |

**Why Separate from Rolling?**

**Before (Problem):**
```
Rolling checkpoints: 014000, 014500, 015000, ..., 019000  (10 checkpoints)
↓ Stop training 10 times (emergency saves)
Emergency saves overwrite rolling: 019100, 019150, 019200, ..., 019450  (10 emergency)
❌ Lost all your periodic snapshots!
```

**After (Solution):**
```
Rolling checkpoints: 014000, 014500, 015000, ..., 019000  (10 rolling)
Emergency checkpoints: 012345_..., 013456_..., 014567_..., ...  (10 emergency)
✅ Both preserved independently!
```

**Recovery Scenario:**
```bash
# Resume from emergency checkpoint
python src/train_improved.py --resume checkpoints/emergency_step_019876_20251106_203145.pt
```

**Use Case:** 
- Recover from unexpected crashes
- Debug issues by seeing when interruptions happened
- Resume after manual stops

---

### 4. 🏆 Best Model Checkpoints

**Purpose:** Keep the best performing models (lowest validation loss).

| File | Description | Updated |
|------|-------------|---------|
| `best_model.pt` | Current best model | When new best found |
| `best_model_backup.pt` | Redundant copy of current best | When new best found |
| `best_model_previous.pt` | Previous best (before replacement) | When new best found |

**Update Flow:**
```
Step 15000: val_loss = 2.5 (new best!)
  1. Copy best_model.pt → best_model_previous.pt  (save old best)
  2. Save new checkpoint → best_model.pt            (save new best)
  3. Copy best_model.pt → best_model_backup.pt      (backup new best)
```

**Why Three Files?**

1. **Current best** (`best_model.pt`): For inference and deployment
2. **Backup** (`best_model_backup.pt`): Corruption protection
3. **Previous** (`best_model_previous.pt`): Rollback if new "best" is actually worse (e.g., validation spike)

**Recovery Scenario:**
```bash
# Use current best
python src/train_improved.py --resume checkpoints/best_model.pt

# If current best is corrupted
python src/train_improved.py --resume checkpoints/best_model_backup.pt

# Rollback to previous best if needed
python src/train_improved.py --resume checkpoints/best_model_previous.pt
```

**Use Case:**
- Inference with best model
- Resume training from best validation performance
- Analysis of model progression

---

### 5. 📅 Epoch Checkpoints

**Purpose:** Mark completion of each epoch for analysis.

| Feature | Details |
|---------|---------|
| **Naming** | `epoch_NNNNN.pt` (5-digit epoch number) |
| **Frequency** | End of each epoch |
| **Retention** | **Keep all** (no cleanup) |
| **Purpose** | Long-term analysis, epoch comparisons |

**Use Case:**
- Compare model performance across epochs
- Resume from specific epoch
- Long-term training analysis

---

## Safety Features

### ⚛️ 1. Atomic Saves

**Problem:** If training crashes mid-save, checkpoint file is corrupted.

**Solution:** Save to temporary file first, validate, then atomic rename.

```python
def _atomic_save(checkpoint, final_path):
    temp_path = final_path + '.tmp'
    
    # 1. Save to temporary file
    torch.save(checkpoint, temp_path)
    
    # 2. Validate it's loadable
    torch.load(temp_path, map_location='cpu')
    
    # 3. Atomic rename (POSIX guarantees atomicity)
    temp_path.rename(final_path)  # ✅ Cannot be interrupted!
```

**Result:** Checkpoints are either complete or don't exist (no partial/corrupted files).

---

### ✅ 2. Automatic Validation

**Problem:** Checkpoint saved successfully but contains corrupted data.

**Solution:** Immediately load checkpoint after saving to verify.

```python
# After saving
try:
    torch.load(temp_path, map_location='cpu', weights_only=False)
    print('✓ Checkpoint validated')
except Exception as e:
    print(f'⚠️  Checkpoint validation failed: {e}')
    # Keep old checkpoint, don't overwrite
```

**Result:** Corrupted checkpoints detected immediately, not hours later.

---

### 🔙 3. Automatic Backup Restoration

**Problem:** Latest checkpoint corrupted, user needs to manually restore.

**Solution:** Automatically restore from backup if save fails.

```python
if not save_successful:
    print('❌ Failed to save latest checkpoint')
    if latest_backup_path.exists() and not latest_path.exists():
        latest_backup_path.rename(latest_path)
        print('  ↻ Restored from backup')
```

**Result:** System automatically recovers from failed saves.

---

### 🗂️ 4. Separate Cleanup Logic

**Rolling Checkpoints:**
```python
def _cleanup_rolling_checkpoints():
    rolling_checkpoints = glob('rolling_step_*.pt')  # Only rolling!
    # Keep last 10, delete older
```

**Emergency Checkpoints:**
```python
def _cleanup_emergency_checkpoints():
    emergency_checkpoints = glob('emergency_step_*.pt')  # Only emergency!
    # Keep last 10, delete older
```

**Result:** Emergency stops don't wipe out rolling checkpoints!

---

### 📦 5. Disk Space Monitoring

**Problem:** Training fills up disk, no warning.

**Solution:** Print disk usage every 10 checkpoint saves.

```
📦 Checkpoint storage: 45.2 GB (10 rolling, 3 emergency, 5 epoch)
```

**Result:** User warned before disk fills up.

---

## Configuration Options

### Command-Line Arguments

```bash
python src/train_improved.py \
    --checkpoint_freq 250 \                      # Save every 250 steps
    --keep_checkpoints 10 \                      # Keep 10 rolling checkpoints
    --keep_emergency_checkpoints 10 \            # Keep 10 emergency checkpoints
    --checkpoint_dir ./checkpoints/              # Checkpoint directory
```

### In Code

```python
checkpoint_manager = CheckpointManager(
    checkpoint_dir='./checkpoints/',
    keep_last_n=10,              # Rolling checkpoints
    keep_emergency_n=10,         # Emergency checkpoints
    master_process=True
)
```

---

## Usage Examples

### Example 1: Normal Training

```bash
python src/train_improved.py --use_tensorboard
```

**What happens:**
- Every 250 steps → save to `rolling_step_*.pt` + `latest.pt`
- End of epoch → save to `epoch_*.pt`
- New best val_loss → save to `best_model*.pt` (3 files)
- Ctrl+C → save to `emergency_step_*_TIMESTAMP.pt`

---

### Example 2: Resume from Latest

```bash
python src/train_improved.py --resume latest --use_tensorboard
```

**Loads:** `checkpoints/latest.pt`

---

### Example 3: Resume from Best

```bash
python src/train_improved.py --resume best --use_tensorboard
```

**Loads:** `checkpoints/best_model.pt`

---

### Example 4: Resume from Specific Checkpoint

```bash
python src/train_improved.py --resume checkpoints/rolling_step_015000.pt
```

**Loads:** Exact checkpoint specified.

---

### Example 5: Generate Safe Resume Command

```bash
# Generate command with exact hyperparameters
python generate_resume_command.py checkpoints/rolling_step_015000.pt

# Compare before changing parameters
python generate_resume_command.py checkpoints/rolling_step_015000.pt \
    --compare \
    --mini_batch_size 64
```

**Output:** Full resume command with all original hyperparameters.

---

### Example 6: List All Checkpoints

```bash
python src/train_improved.py --list_checkpoints
```

**Output:**
```
================================================================================
AVAILABLE CHECKPOINTS
================================================================================

📌 Latest Checkpoints:
--------------------------------------------------------------------------------
  latest.pt:        step=19,000, val_loss=3.2345, size=456.7MB
  latest_backup.pt: step=18,750, val_loss=3.2456, size=456.7MB

🏆 Best Model Checkpoints:
--------------------------------------------------------------------------------
  best_model.pt:          step=17,500, val_loss=3.1234, size=456.7MB
  best_model_backup.pt:   step=17,500, val_loss=3.1234, size=456.7MB
  best_model_previous.pt: step=15,000, val_loss=3.2000, size=456.7MB

🔄 Rolling Checkpoints (10 total, showing last 10):
--------------------------------------------------------------------------------
  rolling_step_014000.pt: step=14,000, val_loss=3.3456, size=456.7MB
  rolling_step_014500.pt: step=14,500, val_loss=3.3123, size=456.7MB
  ...
  rolling_step_019000.pt: step=19,000, val_loss=3.2345, size=456.7MB
  rolling_last_backup.pt: step=19,000, val_loss=3.2345, size=456.7MB (backup)

🚨 Emergency Checkpoints (3 total):
--------------------------------------------------------------------------------
  emergency_step_012345_20251106_143022.pt: step=12,345, saved=2025-11-06 14:30:22
  emergency_step_013456_20251106_150831.pt: step=13,456, saved=2025-11-06 15:08:31
  emergency_step_014567_20251106_161245.pt: step=14,567, saved=2025-11-06 16:12:45

📅 Epoch Checkpoints (2 total):
--------------------------------------------------------------------------------
  epoch_00000.pt: step=19,073, epoch=0, val_loss=3.2123, size=456.7MB
  epoch_00001.pt: step=38,146, epoch=1, val_loss=2.9876, size=456.7MB

📦 Total Storage: 12.45 GB
================================================================================
```

---

## Recovery Scenarios

### Scenario 1: Latest Checkpoint Corrupted

**Error:**
```
RuntimeError: PytorchStreamReader failed reading zip archive
```

**Solution:**
```bash
# Try backup
python src/train_improved.py --resume checkpoints/latest_backup.pt

# Or use most recent rolling
python src/train_improved.py --resume checkpoints/rolling_last_backup.pt
```

---

### Scenario 2: Lost Recent Progress (Want to Go Back)

**Situation:** Latest checkpoint at step 19,000, but you want to resume from 15,000.

**Solution:**
```bash
# List available rolling checkpoints
python src/train_improved.py --list_checkpoints

# Resume from desired checkpoint
python src/train_improved.py --resume checkpoints/rolling_step_015000.pt
```

---

### Scenario 3: Training Stopped Multiple Times

**Situation:** Stopped training 10+ times for testing.

**Before (old system):** Regular checkpoints wiped out by emergency saves ❌

**After (new system):** Both preserved! ✅

```bash
# See both rolling and emergency checkpoints
python src/train_improved.py --list_checkpoints

# Resume from any of them
python src/train_improved.py --resume checkpoints/rolling_step_018000.pt
# OR
python src/train_improved.py --resume checkpoints/emergency_step_019123_20251106_203145.pt
```

---

### Scenario 4: Validation Spike (Bad "Best" Model)

**Situation:** New "best" model saved, but validation was actually a spike.

**Solution:**
```bash
# Rollback to previous best
python src/train_improved.py --resume checkpoints/best_model_previous.pt
```

---

### Scenario 5: Checkpoint Saved During Crash

**Before (old system):** Corrupted checkpoint, training lost ❌

**After (new system):** 
1. Atomic save ensures checkpoint is complete OR doesn't exist
2. Validation catches corruption immediately
3. Backup automatically restored

**Result:** Training never lost! ✅

---

## Comparison: Old vs New System

| Feature | Old System | New System |
|---------|-----------|------------|
| **Atomic saves** | ❌ No | ✅ Yes |
| **Validation** | ❌ No | ✅ Yes (immediate) |
| **Emergency vs Rolling** | ❌ Same pool | ✅ Separate (10 each) |
| **Best model backups** | ⚠️ 1 file | ✅ 3 files (current + backup + previous) |
| **Rolling backup** | ❌ No | ✅ Yes (`rolling_last_backup.pt`) |
| **Timestamps** | ❌ No | ✅ Yes (emergency only) |
| **Disk monitoring** | ❌ No | ✅ Yes (every 10 saves) |
| **Corruption recovery** | ⚠️ Manual | ✅ Automatic |
| **Resume safety** | ⚠️ Manual | ✅ `generate_resume_command.py` |

---

## Best Practices

### ✅ DO:

1. **Use the resume command generator**
   ```bash
   python generate_resume_command.py checkpoints/latest.pt
   ```

2. **Regularly check disk usage**
   ```bash
   du -sh checkpoints/
   ```

3. **Test checkpoint recovery before long training**
   ```bash
   # Train for a few steps
   python src/train_improved.py --num_epochs 1 --use_tensorboard
   # Stop with Ctrl+C (tests emergency save)
   # Resume
   python src/train_improved.py --resume latest --num_epochs 1
   ```

4. **Keep epoch checkpoints for analysis**
   - They're saved once per epoch, minimal disk usage
   - Useful for comparing epoch-to-epoch progress

5. **Use separate checkpoint directory for experiments**
   ```bash
   python src/train_improved.py --checkpoint_dir ./checkpoints_experiment1/
   ```

---

### ❌ DON'T:

1. **Don't manually delete checkpoints while training**
   - Let automatic cleanup handle it
   - If disk space critical, increase `--keep_checkpoints` value

2. **Don't mix checkpoint directories**
   - Use different `--checkpoint_dir` for different experiments
   - Don't copy checkpoints between directories

3. **Don't resume with drastically different hyperparameters**
   - Use `generate_resume_command.py --compare` first
   - Optimizer state was built with original parameters

4. **Don't ignore validation failures**
   ```
   ⚠️  Checkpoint validation failed: <error>
   ```
   - This means checkpoint is corrupted
   - Use backup or previous checkpoint

---

## Troubleshooting

### Issue: "Checkpoint validation failed"

**Cause:** Checkpoint corrupted during save.

**Solution:** System automatically keeps old checkpoint. Check disk space.

---

### Issue: Disk full during training

**Prevention:** Monitor disk usage output every 10 saves.

**Solution:**
```bash
# Clean up old experiment checkpoints
rm -rf checkpoints_old/

# Or increase cleanup frequency
python src/train_improved.py --keep_checkpoints 5  # Keep fewer rolling
```

---

### Issue: Emergency checkpoint not created on Ctrl+C

**Check:**
1. Are you running as master process? (rank 0)
2. Did training actually start? (past initialization)
3. Is checkpoint directory writable?

**Debug:**
```bash
# Check permissions
ls -la checkpoints/

# Check disk space
df -h .
```

---

### Issue: Too many checkpoint files

**Normal:** You should see:
- 1 latest + 1 backup
- 10 rolling + 1 rolling backup
- Up to 10 emergency
- 3 best model files
- N epoch files (one per completed epoch)

**Total:** ~26 files max (excluding epochs)

**If more:** Cleanup may have failed. Manually delete oldest:
```bash
ls -lt checkpoints/rolling_step_*.pt | tail -n +11 | awk '{print $NF}' | xargs rm
```

---

## Technical Details

### Checkpoint Contents

Every checkpoint contains:
```python
{
    'step': 15000,                        # Training step
    'epoch': 0,                           # Current epoch
    'model': model.state_dict(),          # Model weights
    'optimizer': optimizer.state_dict(),  # Optimizer state
    'train_loss': 3.4567,                 # Training loss
    'val_loss': 3.2345,                   # Validation loss
    'rng_state': {                        # For reproducibility
        'python': <state>,
        'numpy': <state>,
        'torch': <state>,
        'torch_cuda': <state>
    },
    'dataloader_state': {                 # Data position
        'train_curr_shard': 42,
        'train_curr_pos': 123456,
        'val_curr_shard': 5,
        'val_curr_pos': 67890
    },
    'args': {...},                        # All hyperparameters
    'timestamp': 1699288234.567,          # Unix timestamp
    'checkpoint_type': 'regular' or 'emergency'
}
```

### File Size

Typical checkpoint size: ~460 MB
- Model weights: ~360 MB
- Optimizer state: ~100 MB
- Other metadata: <1 MB

---

## See Also

- `CHECKPOINTING_GUIDE.md` - Original checkpointing documentation
- `RESUME_COMMAND_GENERATOR.md` - Safe resume command generation
- `TENSORBOARD_GUIDE.md` - Training monitoring
- `PYTORCH_2.6_FIX.md` - PyTorch 2.6+ compatibility fixes

---

**Created:** 2025-11-06  
**Status:** ✅ Production Ready  
**Tested:** Yes (atomic saves, validation, recovery scenarios)


