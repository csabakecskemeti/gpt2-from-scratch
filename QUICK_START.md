# Quick Start Guide - Improved Training with Checkpointing

## 🚀 TL;DR - What You Need to Know

Your training now has **smart checkpointing** that saves your progress frequently and lets you resume if anything goes wrong.

### If Your Training Crashes/Stops:

```bash
# Resume from where you left off (simple!)
python src/train_improved.py --resume latest

# Or with multi-GPU
torchrun --standalone --nproc_per_node=4 src/train_improved.py --resume latest
```

### To Stop Training Safely:

Press **Ctrl+C** - it will save a checkpoint before exiting automatically!

---

## 📋 Quick Command Reference

| Task | Command |
|------|---------|
| **Start new training** | `python src/train_improved.py` |
| **Resume training** | `python src/train_improved.py --resume latest` |
| **List all checkpoints** | `python src/train_improved.py --list_checkpoints` |
| **Start with 4 GPUs** | `torchrun --standalone --nproc_per_node=4 src/train_improved.py` |
| **Resume with 4 GPUs** | `torchrun --standalone --nproc_per_node=4 src/train_improved.py --resume latest` |

---

## 🎯 Key Improvements

### Before (`train.py`):
- ❌ Saves checkpoint every 10,000 steps only
- ❌ Can't resume training properly
- ❌ If crash → lose hours of training
- ❌ No automatic cleanup → fills disk

### Now (`train_improved.py`):
- ✅ Saves checkpoint every 250 steps (~1 minute)
- ✅ Full resume capability
- ✅ If crash → lose at most 1 minute
- ✅ Automatic cleanup → keeps only recent checkpoints
- ✅ Tracks best model automatically
- ✅ Saves one checkpoint per epoch
- ✅ Graceful shutdown on Ctrl+C

---

## 💾 Where Are Checkpoints Saved?

By default: `./checkpoints/`

**Important checkpoints:**
- `checkpoints/latest.pt` - Most recent (use this to resume)
- `checkpoints/best_model.pt` - Best validation loss
- `checkpoints/epoch_00001.pt` - End of epoch 1
- `checkpoints/rolling_step_XXXXXX.pt` - Recent checkpoints (keeps last 10)

---

## 🔧 Typical Workflow

### 1. Start Training

```bash
# Single GPU
python src/train_improved.py

# 4 GPUs (your DGX setup)
torchrun --standalone --nproc_per_node=4 src/train_improved.py
```

### 2. Training Output

You'll see:
```
step  250 | epoch 0 | loss: 3.456789 | lr: 6.00e-04 | dt: 234.56ms | tok/sec: 2234.56
✓ Saved latest checkpoint: checkpoints/latest.pt
✓ Saved rolling checkpoint: checkpoints/rolling_step_000250.pt
```

### 3. If Training Stops (Crash/Power Loss/Ctrl+C)

Simply resume:
```bash
python src/train_improved.py --resume latest
```

Output will show:
```
============================================================
Loading checkpoint from: checkpoints/latest.pt
============================================================
✓ Restored model and optimizer state
✓ Restored RNG states
✓ Restored data loader state

Resuming from:
  - Step: 5000
  - Epoch: 1
  - Train Loss: 3.234567
  - Val Loss: 3.345678
============================================================

Starting training from step 5001 to 95365
```

### 4. Check Available Checkpoints Anytime

```bash
./list_checkpoints.sh

# Or
python src/train_improved.py --list_checkpoints
```

---

## ⚙️ Common Configurations

### Save More Frequently (Paranoid Mode)

```bash
python src/train_improved.py --checkpoint_freq 100  # Every 100 steps
```

### Save Less Frequently (Disk Space Constrained)

```bash
python src/train_improved.py --checkpoint_freq 500 --keep_checkpoints 5
```

### Use Custom Checkpoint Directory

```bash
python src/train_improved.py --checkpoint_dir /path/to/fast/ssd/checkpoints/
```

---

## 🆘 Emergency Procedures

### Training Crashed - What to Do?

1. **Check if latest checkpoint exists:**
   ```bash
   ls -lh checkpoints/latest.pt
   ```

2. **Resume from latest:**
   ```bash
   python src/train_improved.py --resume latest
   ```

3. **If latest is corrupted, try backup:**
   ```bash
   python src/train_improved.py --resume checkpoints/latest_backup.pt
   ```

4. **If that fails, use a rolling checkpoint:**
   ```bash
   # List available
   ls -lh checkpoints/rolling_step_*.pt
   
   # Resume from specific one
   python src/train_improved.py --resume checkpoints/rolling_step_005000.pt
   ```

### Want to Go Back to Earlier Checkpoint

```bash
# List all checkpoints with their validation losses
python src/train_improved.py --list_checkpoints

# Resume from specific checkpoint
python src/train_improved.py --resume checkpoints/rolling_step_004500.pt
```

### Out of Disk Space

```bash
# Check current usage
du -sh checkpoints/

# Reduce number of rolling checkpoints kept
python src/train_improved.py --keep_checkpoints 5

# Manually delete old epoch checkpoints you don't need
rm checkpoints/epoch_00001.pt
```

---

## 📊 Monitoring Progress

### During Training

Watch for these lines:
```
step  5000 | epoch 1 | loss: 3.234567 | lr: 6.00e-04 | ...
Val loss: 3.345678
✓ Saved latest checkpoint: checkpoints/latest.pt
✓ Saved rolling checkpoint: checkpoints/rolling_step_005000.pt
```

### Check Best Model Performance

```bash
python -c "import torch; ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu'); print(f'Best Val Loss: {ckpt[\"val_loss\"]:.4f} at step {ckpt[\"step\"]}')"
```

### Check Training Progress

```bash
python -c "import torch; ckpt = torch.load('checkpoints/latest.pt', map_location='cpu'); print(f'Step: {ckpt[\"step\"]}/{95365} ({100*ckpt[\"step\"]/95365:.1f}%), Epoch: {ckpt[\"epoch\"]}/5, Val Loss: {ckpt[\"val_loss\"]:.4f}')"
```

---

## 🎓 Understanding Checkpoint Types

### Latest Checkpoint
- **File:** `checkpoints/latest.pt`
- **When:** Every 250 steps (overwrites)
- **Use for:** Resuming after crash/stop
- **Kept:** Only 1 (plus 1 backup)

### Rolling Checkpoints
- **Files:** `checkpoints/rolling_step_XXXXXX.pt`
- **When:** Every 250 steps
- **Use for:** Going back to recent point
- **Kept:** Last 10

### Epoch Checkpoints
- **Files:** `checkpoints/epoch_XXXXX.pt`
- **When:** End of each epoch
- **Use for:** Major milestones
- **Kept:** All (5 total for 5 epochs)

### Best Model
- **File:** `checkpoints/best_model.pt`
- **When:** Whenever validation loss improves
- **Use for:** Deployment, inference
- **Kept:** Only 1 (best so far)

---

## 💡 Pro Tips

### 1. Test Resume Early

Don't wait until you need it! Test within first 10 minutes:

```bash
# Start training
python src/train_improved.py

# Wait 5-10 minutes, then Ctrl+C

# Resume
python src/train_improved.py --resume latest

# Verify it continues smoothly
```

### 2. Backup Best Model Periodically

```bash
# Every few days, backup to safe location
cp checkpoints/best_model.pt ~/backups/best_model_day3.pt
```

### 3. Monitor Disk Usage

```bash
# Add to cron or check daily
du -sh checkpoints/
```

### 4. Compare Checkpoints

```bash
# See which checkpoint has best validation loss
for f in checkpoints/*.pt; do
    echo -n "$f: "
    python -c "import torch; print(torch.load('$f', map_location='cpu')['val_loss'])"
done | sort -t: -k2 -n
```

---

## 🔄 Migrating from Original Script

If you started training with `train.py` already:

### Option 1: Start Fresh (Recommended)

The old checkpoints are incomplete. Better to restart:

```bash
# Backup old logs/checkpoints
mv logs logs_old
mv src/logs src/logs_old

# Start fresh with improved script
python src/train_improved.py
```

### Option 2: Continue Training

If you've already trained significantly:

1. Note your current step (check logs)
2. Start improved training fresh
3. Accept you might repeat some training (worth it for safety!)

---

## ❓ FAQ

**Q: How much disk space will checkpoints use?**

A: ~9 GB total (10 rolling × 500MB + 5 epoch × 500MB + 2 latest × 500MB + 1 best × 500MB)

**Q: Will checkpointing slow down training?**

A: Minimal impact (<5%). Saving 500MB takes ~1-2 seconds every 250 steps (~50 seconds).

**Q: Can I change hyperparameters when resuming?**

A: Yes! Most hyperparameters can be changed (learning rate, eval frequency, etc.). The script will use new values.

**Q: What if I lose the latest checkpoint?**

A: Use `latest_backup.pt` or any rolling checkpoint.

**Q: How do I know which checkpoint is best?**

A: Check `checkpoints/best_model.pt` - it's automatically tracked!

**Q: Can I resume with different number of GPUs?**

A: Not recommended. Resume with same GPU count as original training.

**Q: What happens on Ctrl+C?**

A: The script catches the signal, saves a checkpoint, then exits cleanly. You can resume immediately.

---

## 🎉 You're Ready!

Key takeaways:
1. ✅ Training is now much safer (saves every 250 steps)
2. ✅ Resume anytime with `--resume latest`
3. ✅ Ctrl+C saves checkpoint before exiting
4. ✅ Best model is tracked automatically
5. ✅ Disk usage is managed automatically

**For your 34-day training, this is essential!** You now have peace of mind that crashes, power outages, or interruptions won't cost you significant training time.

Start training with confidence! 🚀

```bash
torchrun --standalone --nproc_per_node=4 src/train_improved.py
```

For detailed information, see `CHECKPOINTING_GUIDE.md`.

