# Start 4-GPU Training - Quick Guide

**Date:** November 3, 2024

---

## ✅ What's Ready

1. ✅ **PyTorch 2.9 Resume Fix** - Applied and tested
2. ✅ **Dataset Shuffling** - Just added (automatic between epochs)
3. ✅ **Checkpointing** - Working perfectly
4. ✅ **TensorBoard** - Ready for monitoring
5. ✅ **Checkpoint at step 329** - Ready to resume from

---

## 🚀 Start Training Command

```bash
cd /home/kecso/Documents/workspace/training/gpt-2/gpt2-from-scratch
source .venv/bin/activate

# Terminal 1: Start 4-GPU training with resume + TensorBoard + Shuffling
torchrun --standalone --nproc_per_node=4 src/train_improved.py \
  --resume latest \
  --use_tensorboard
```

## 📊 Start TensorBoard (Separate Terminal)

```bash
cd /home/kecso/Documents/workspace/training/gpt-2/gpt2-from-scratch
source .venv/bin/activate
./start_tensorboard.sh

# Or manually:
tensorboard --logdir=runs --bind_all
```

Then open: **http://localhost:6006**

---

## ⚡ Expected Performance

### Before (1 GPU - what you were running)
- **Step time:** ~37.5 seconds
- **Total time:** ~41 days
- **Tokens/sec:** ~14,000
- **Progress:** 0.34% (329/95,365 steps)

### After (4 GPUs - what you'll get now)
- **Step time:** ~9-10 seconds (4x faster!)
- **Total time:** **~10.4 days** 🎉
- **Tokens/sec:** ~56,000
- **Time saved:** ~30 days!

---

## 🔀 What You'll See

### At Startup
```
============================================================
Loading checkpoint from: checkpoints/latest.pt
============================================================
Resuming from:
  - Step: 329
  - Epoch: 0
...
GPU: 0, 1, 2, 3  ← All 4 GPUs!
```

### At Epoch Boundary (step 19073)
```
============================================================
🎯 Completed Epoch 0
============================================================

🔀 Shuffled 99 training shards for new epoch  ← NEW!
🔄 Starting Epoch 1

step 19073 | epoch 1 | loss: 3.456 | lr: 6.0e-04 | ...
```

### During Training
```
step 19074 | epoch 1 | loss: 3.445 | lr: 6.0e-04 | norm: 0.85 | dt: 9500ms | tok/sec: 55000
                                                                  ↑            ↑
                                                           4x faster!    4x throughput!
```

---

## 🎯 Training Timeline

**Current:** Step 329 (0.34% complete)

**Milestones with 4 GPUs:**

| Step | Epoch | Event | Time from Now |
|------|-------|-------|---------------|
| 329 | 0 | **← You are here** | Now |
| 250 | 0 | Checkpoint saved | -2.6 hours (past) |
| 500 | 0 | Next checkpoint | ~27 minutes |
| 750 | 0 | Checkpoint | ~1.9 hours |
| 1000 | 0 | Checkpoint | ~3.1 hours |
| 19073 | 0→1 | **First shuffle!** | ~2.2 days |
| 38146 | 1→2 | Second shuffle | ~4.4 days |
| 57219 | 2→3 | Third shuffle | ~6.6 days |
| 76292 | 3→4 | Fourth shuffle | ~8.8 days |
| 95365 | 4→5 | **Training complete!** | **~10.4 days** |

---

## 📈 What Dataset Shuffling Does

### At Epoch Boundaries

**Epoch 0 (now):** Shards in original order
- `[shard_000.npy, shard_001.npy, shard_002.npy, ...]`

**Epoch 1 (step 19073):** Shards shuffled!
- `[shard_042.npy, shard_007.npy, shard_091.npy, ...]`

**Epoch 2 (step 38146):** Different shuffle
- `[shard_023.npy, shard_088.npy, shard_003.npy, ...]`

### Benefits
- ✅ Better generalization
- ✅ Prevents order bias
- ✅ Standard ML practice
- ✅ Automatic (no action needed)
- ✅ Zero performance cost

---

## 📝 What's Been Fixed/Added Today

### 1. PyTorch 2.9 Resume Fix
**Problem:** `_pickle.UnpicklingError` and RNG state device mismatch

**Solution:**
- Added `weights_only=False` to all `torch.load()` calls
- Added CPU conversion for RNG states

**Status:** ✅ Fixed and tested

**Doc:** `PYTORCH_2.6_FIX.md`

### 2. Dataset Shuffling
**Problem:** "Future work" item in README

**Solution:**
- Added `shuffle_shards()` method to dataloader
- Automatic shuffling at epoch boundaries
- Uses numpy RNG (saved in checkpoints)

**Status:** ✅ Implemented and ready

**Doc:** `DATASET_SHUFFLING_FEATURE.md`

### 3. Training Time Calculation
**Problem:** Miscalculated training time

**Reality:**
- Single GPU: ~41 days (not 34)
- 4 GPUs: ~10 days

**Status:** ✅ Calculated correctly

**Script:** `estimate_training_time.py`

---

## 🛡️ Safety Features

### Checkpointing
- ✅ Saves every 250 steps (~2.6 hours at 4-GPU speed)
- ✅ Keeps last 10 rolling checkpoints
- ✅ Best model tracked automatically
- ✅ Ctrl+C saves emergency checkpoint

### Resume Capability
- ✅ Can resume from any checkpoint
- ✅ Exact training state restored
- ✅ RNG states preserved (reproducible)
- ✅ Shuffle order maintained

### Monitoring
- ✅ TensorBoard real-time metrics
- ✅ Text logs in `logs/log.txt`
- ✅ Generated text samples
- ✅ HellaSwag accuracy tracking

---

## 🔧 Troubleshooting

### Training Crashes
```bash
# Just resume - automatic
torchrun --standalone --nproc_per_node=4 src/train_improved.py --resume latest --use_tensorboard
```

### GPU Out of Memory
```bash
# Reduce batch size (if needed)
torchrun --standalone --nproc_per_node=4 src/train_improved.py \
  --resume latest \
  --use_tensorboard \
  --mini_batch_size 16
```

### Check Progress
```bash
# List checkpoints
python src/train_improved.py --list_checkpoints

# Watch logs
tail -f logs/log.txt

# Calculate time remaining
python estimate_training_time.py
```

---

## 📊 Monitoring Checklist

### Terminal
- [ ] Training running in Terminal 1
- [ ] TensorBoard running in Terminal 2
- [ ] No error messages
- [ ] Step time ~9-10 seconds

### TensorBoard (localhost:6006)
- [ ] Loss curves decreasing
- [ ] Learning rate following schedule
- [ ] Tokens/sec ~55,000
- [ ] All 4 GPUs utilized

### Checkpoints
- [ ] Saved every 250 steps
- [ ] File size ~500 MB each
- [ ] Latest checkpoint exists

---

## 🎉 You're Ready!

Everything is set up and ready to go:

✅ Code fixed (PyTorch 2.9 compatibility)  
✅ Shuffling added (better generalization)  
✅ 4 GPUs configured (4x faster)  
✅ Checkpoint ready (resume from step 329)  
✅ TensorBoard ready (monitoring)  

**Just run:**

```bash
torchrun --standalone --nproc_per_node=4 src/train_improved.py --resume latest --use_tensorboard
```

**ETA to completion:** ~10.4 days 🚀

---

**Good luck with your training!** 🎯

Check TensorBoard frequently to monitor progress and enjoy the 4x speedup!

