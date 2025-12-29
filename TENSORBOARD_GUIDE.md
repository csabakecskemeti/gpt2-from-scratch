# TensorBoard Monitoring Guide

## 🎯 Overview

TensorBoard is now integrated into `train_improved.py` and provides **real-time monitoring** of your training progress through beautiful visualizations.

---

## 🚀 Quick Start

### Enable TensorBoard Logging

Simply add the `--use_tensorboard` flag:

```bash
# Single GPU
python src/train_improved.py --use_tensorboard

# Multi-GPU
torchrun --standalone --nproc_per_node=4 src/train_improved.py --use_tensorboard
```

### View TensorBoard Dashboard

**In a separate terminal** (while training is running):

```bash
tensorboard --logdir=runs --bind_all
```

Then open in your browser: **http://localhost:6006**

---

## 📊 What Gets Logged

### Scalars (Every Step)

**Loss Metrics:**
- `Loss/train` - Training loss
- `Loss/validation` - Validation loss

**Learning Rate:**
- `Learning_Rate/lr` - Current learning rate (shows warmup & decay)

**Gradients:**
- `Gradient/norm` - Gradient L2 norm (after clipping)

**Performance:**
- `Performance/step_time_ms` - Time per training step (ms)
- `Performance/tokens_per_sec` - Training throughput

**Training Progress:**
- `Training/epoch` - Current epoch number

**Evaluation:**
- `Evaluation/hellaswag_accuracy` - Common-sense reasoning accuracy

### Histograms (Every 1000 Steps)

**Parameters:**
- `Parameters/*` - Model weight distributions (per layer)

**Gradients:**
- `Gradients/*` - Gradient distributions (per layer)

**Use case:** Detect vanishing/exploding gradients, monitor weight updates

### Text (Every eval_freq Steps)

**Generated Samples:**
- `Generated_Samples/text` - Model-generated text samples

**Use case:** Qualitatively assess model progress

### Hyperparameters

**On startup:**
- All training hyperparameters logged to HParams tab
- Compare different runs easily

---

## 🔧 Configuration Options

### Custom Run Name

```bash
python src/train_improved.py --use_tensorboard --run_name "gpt2_lr1e-3_bs32"
```

### Custom TensorBoard Directory

```bash
python src/train_improved.py --use_tensorboard --tensorboard_dir /path/to/tensorboard/logs
```

### Full Example

```bash
torchrun --standalone --nproc_per_node=4 src/train_improved.py \
  --use_tensorboard \
  --run_name "gpt2_5epoch_run1" \
  --tensorboard_dir ./tensorboard_logs \
  --checkpoint_freq 250 \
  --eval_freq 250
```

---

## 🌐 Remote Access (SSH Tunnel)

If training on a remote server (like your DGX), use SSH port forwarding:

### Method 1: SSH Tunnel (Recommended)

**On your local machine:**

```bash
ssh -L 6006:localhost:6006 user@dgx-spark-hostname
```

Then on the **remote server (DGX)**:

```bash
tensorboard --logdir=runs --bind_all --port 6006
```

Finally, open on **your local machine**: **http://localhost:6006**

### Method 2: Bind to All Interfaces

**On remote server (DGX):**

```bash
tensorboard --logdir=runs --bind_all --port 6006
```

Then access directly: **http://dgx-spark-hostname:6006**

⚠️ **Security note:** This exposes TensorBoard to network. Use SSH tunnel for better security.

---

## 📈 Dashboard Overview

### Scalars Tab

This is where you'll spend most time!

**Key Charts to Monitor:**

1. **Loss/train & Loss/validation**
   - Should steadily decrease
   - Val loss should track train loss
   - If val >> train: overfitting

2. **Learning_Rate/lr**
   - Verify warmup & cosine decay schedule
   - Should increase linearly during warmup
   - Then smoothly decay

3. **Gradient/norm**
   - Should be stable (clipped at 1.0)
   - Spikes indicate instability
   - Consistently high (near 1.0) = aggressive learning

4. **Performance/tokens_per_sec**
   - Monitor training efficiency
   - Drops may indicate bottlenecks
   - Should be consistent throughout training

5. **Evaluation/hellaswag_accuracy**
   - Measures reasoning ability
   - Should improve over time
   - Random baseline: 0.25 (25%)

### Distributions/Histograms Tab

**Parameters:**
- Watch for layers not updating (flat distribution)
- Ensure weights don't become too large/small

**Gradients:**
- Watch for vanishing gradients (values near zero)
- Watch for exploding gradients (before clipping)

### Text Tab

**Generated Samples:**
- Early training: Gibberish
- Mid training: Grammatical but nonsensical
- Late training: Coherent and contextual

### HParams Tab

**Compare different runs:**
- See which hyperparameters work best
- Sort by validation loss
- Identify optimal configurations

---

## 💡 TensorBoard Tips & Tricks

### Smooth Noisy Curves

In TensorBoard UI, use the **smoothing slider** (top left):
- Slide right to smooth out noise
- See overall trends more clearly

### Compare Multiple Runs

```bash
# Run 1
python src/train_improved.py --use_tensorboard --run_name "run1_lr1e-3"

# Run 2 (in parallel or later)
python src/train_improved.py --use_tensorboard --run_name "run2_lr5e-4"

# View both
tensorboard --logdir=runs
```

In TensorBoard, select both runs in the left sidebar!

### Filter Charts

Use the search box to filter by metric name:
- Type "loss" to see only loss charts
- Type "gradient" to see gradient metrics

### Download Data

Click the **⋮** (three dots) on any chart → **Download CSV**

Useful for creating publication-quality plots in matplotlib/R.

### Refresh Interval

TensorBoard auto-refreshes every 30 seconds. To manually refresh:
- Click the reload icon (⟳) in top right

---

## 🔍 Interpreting Your Metrics

### Healthy Training

✅ **Training loss:** Steadily decreasing  
✅ **Validation loss:** Tracking train loss (slightly higher)  
✅ **Gradient norm:** Stable, often near 0.5-1.0  
✅ **Throughput:** Consistent  
✅ **HellaSwag:** Gradually improving  
✅ **Generated text:** Getting more coherent  

### Warning Signs

⚠️ **Val loss >> train loss:** Overfitting (consider more data, regularization)  
⚠️ **Loss not decreasing:** LR too low, or bug in code  
⚠️ **Loss exploding (NaN):** LR too high, numerical instability  
⚠️ **Gradient norm consistently 1.0:** Gradients being clipped frequently (very aggressive learning)  
⚠️ **Throughput dropping:** Bottleneck (CPU, I/O, memory)  
⚠️ **Flat HellaSwag:** Model not learning reasoning  

### Example Timeline (GPT-2 Small, 5 Epochs)

| Step | Train Loss | Val Loss | HellaSwag | Generated Text Quality |
|------|------------|----------|-----------|------------------------|
| 0 | ~10.5 | ~10.5 | 0.25 | Random characters |
| 5,000 | ~4.5 | ~4.7 | 0.26 | Words, no grammar |
| 20,000 | ~3.5 | ~3.7 | 0.28 | Short phrases |
| 50,000 | ~3.0 | ~3.2 | 0.32 | Grammatical sentences |
| 95,365 | ~2.8 | ~3.0 | 0.35 | Coherent paragraphs |

---

## 📸 Screenshot Examples

### Loss Curves

You should see:
- Blue line (train loss) decreasing smoothly
- Orange line (val loss) tracking above it
- Both following cosine decay of learning rate

### Learning Rate Schedule

You should see:
- Linear increase from 0 to max_lr (warmup)
- Smooth cosine decay from max_lr to min_lr
- Matches the schedule formula

### Gradient Norm

You should see:
- Most values between 0.3-1.0
- Clipped at 1.0 (horizontal line)
- Occasional spikes (normal)

---

## 🛠️ Troubleshooting

### Problem: "No dashboards are active for the current data set"

**Cause:** TensorBoard started before any logs were written

**Solution:**
```bash
# Wait a few minutes for training to write first logs
# Or restart TensorBoard:
Ctrl+C
tensorboard --logdir=runs
```

### Problem: Can't access TensorBoard remotely

**Check:**
1. Is TensorBoard running? `ps aux | grep tensorboard`
2. Is port 6006 open? `netstat -tuln | grep 6006`
3. Try SSH tunnel method instead

### Problem: TensorBoard shows old data

**Solution:**
```bash
# Clear cache
rm -rf runs/.tensorboard-*
# Restart TensorBoard
```

### Problem: Disk filling up with TensorBoard logs

**Cause:** Histograms take up space (logged every 1000 steps)

**Solution 1:** Delete old runs
```bash
rm -rf runs/old_run_name
```

**Solution 2:** Reduce histogram frequency (edit train_improved.py line 405)
```python
if step % 5000 == 0:  # Changed from 1000 to 5000
```

### Problem: Port 6006 already in use

**Solution:**
```bash
# Use different port
tensorboard --logdir=runs --port 6007

# Or kill existing TensorBoard
pkill -f tensorboard
```

---

## 📊 Disk Usage

TensorBoard logs use disk space:

**Scalars:** ~1 KB per step  
**Histograms:** ~500 KB per step (logged every 1000 steps)  
**Text:** ~10 KB per log  

**Estimated for 95K step training:**
- Scalars: 95K × 1 KB = ~95 MB
- Histograms: 95 steps × 500 KB = ~47.5 MB
- Text: ~400 logs × 10 KB = ~4 MB
- **Total: ~150 MB per run**

Much smaller than model checkpoints (~500 MB each)!

---

## 🎨 Customizing TensorBoard

### Dark Mode

TensorBoard automatically uses dark mode if your browser/OS is set to dark mode!

### Custom Plugins

TensorBoard supports plugins for 3D visualizations, audio, etc. (advanced)

---

## 📋 Command Reference

| Task | Command |
|------|---------|
| **Start training with TB** | `python src/train_improved.py --use_tensorboard` |
| **View TensorBoard** | `tensorboard --logdir=runs` |
| **Custom run name** | `--run_name "my_experiment"` |
| **Custom TB directory** | `--tensorboard_dir ./tb_logs` |
| **SSH tunnel** | `ssh -L 6006:localhost:6006 user@host` |
| **Custom port** | `tensorboard --logdir=runs --port 6007` |
| **Bind all IPs** | `tensorboard --logdir=runs --bind_all` |

---

## 🔄 TensorBoard + Checkpointing

Both work together seamlessly!

**Scenario:** Training crashes at step 50,000

**Resume:**
```bash
python src/train_improved.py --resume latest --use_tensorboard --run_name "gpt2_resumed"
```

**TensorBoard will:**
- ✅ Continue logging from step 50,001
- ✅ Show complete timeline (if same run_name)
- ✅ Or create new timeline (if different run_name)

---

## 🎯 Recommended Workflow

### 1. Start Training

```bash
torchrun --standalone --nproc_per_node=4 src/train_improved.py \
  --use_tensorboard \
  --run_name "gpt2_production_run" \
  --checkpoint_freq 250 \
  --eval_freq 250
```

### 2. Start TensorBoard (in separate terminal)

```bash
tensorboard --logdir=runs --bind_all
```

### 3. Set Up SSH Tunnel (if remote)

**On your local machine:**
```bash
ssh -L 6006:localhost:6006 user@dgx-spark
```

### 4. Open Dashboard

Open browser: **http://localhost:6006**

### 5. Monitor

Check dashboard every few hours:
- Verify loss is decreasing
- Check for any anomalies
- Review generated text quality
- Monitor throughput

### 6. Compare Runs

After trying different hyperparameters, compare in TensorBoard:
- Select multiple runs in sidebar
- See which performs best

---

## 📱 Mobile Monitoring

TensorBoard works on mobile browsers!

**On phone/tablet:**
1. Connect to same network (or VPN)
2. Open browser
3. Navigate to TensorBoard URL
4. Pinch to zoom on graphs

Great for checking training progress on the go!

---

## 🎓 Pro Tips

### Tip 1: Bookmark Your TensorBoard URL

If using SSH tunnel, bookmark `http://localhost:6006` for quick access.

### Tip 2: Use tmux/screen for Long-Running TensorBoard

```bash
# Start tmux session
tmux new -s tensorboard

# Inside tmux
tensorboard --logdir=runs --bind_all

# Detach: Ctrl+B then D
# Reattach: tmux attach -t tensorboard
```

### Tip 3: Compare Against Baselines

Keep a "baseline" run for comparison:
```bash
# Never delete runs/baseline_gpt2
```

### Tip 4: Take Screenshots

Use TensorBoard's screenshot feature (📷 icon) to document progress!

### Tip 5: Export for Papers/Presentations

Download CSV data → Create publication-quality plots in your favorite tool.

---

## ✅ Quick Checklist

Before starting your 34-day training:

- [ ] TensorBoard installed: `pip install tensorboard`
- [ ] Training starts with `--use_tensorboard`
- [ ] TensorBoard dashboard accessible
- [ ] SSH tunnel working (if remote)
- [ ] Bookmarked TensorBoard URL
- [ ] Tested on phone/tablet (optional)
- [ ] Know how to interpret key metrics
- [ ] Set up daily check reminder

---

## 🎉 You're Ready!

TensorBoard is now fully integrated with your training!

**Start training:**
```bash
torchrun --standalone --nproc_per_node=4 src/train_improved.py --use_tensorboard
```

**View progress:**
```bash
tensorboard --logdir=runs
```

**Enjoy beautiful real-time monitoring of your 34-day training run!** 📊✨

For detailed training setup, see `QUICK_START.md`.

