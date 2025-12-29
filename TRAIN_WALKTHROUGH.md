# Training Script Walkthrough

This document provides a comprehensive walkthrough of the `train.py` script, which implements the complete training pipeline for GPT-2, including distributed training, evaluation, and text generation.

## Overview

The training script orchestrates:
- **Model initialization**: Sets up GPT-2 model
- **Distributed training**: Multi-GPU support with DDP (Distributed Data Parallel)
- **Training loop**: Forward/backward passes with gradient accumulation
- **Validation**: Periodic loss evaluation
- **HellaSwag evaluation**: Tests common-sense reasoning
- **Text generation**: Samples text to verify model behavior
- **Checkpointing**: Saves model at intervals

---

## File Structure

The script is organized into:
1. **Imports and configuration** (lines 1-22)
2. **`Trainer` class** (lines 24-254): Main training logic
3. **`GPTConfig` dataclass** (lines 257-264): Model hyperparameters
4. **`get_args()` function** (lines 266-285): Command-line arguments
5. **`main()` function** (lines 288-392): Entry point and setup

---

## Part 1: Imports and Configuration (lines 1-22)

```python
import os
import math
import numpy as np
import time
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
```

**Key imports:**

- **`tiktoken`**: OpenAI's tokenizer (encodes/decodes text)
- **`torch.distributed`**: Multi-GPU communication
- **`DDP`**: Wraps model for distributed training
- **`dataclasses`**: Clean way to define configuration

```python
torch.set_float32_matmul_precision('high')    # enable TF32 precision
```

**TF32 precision:** Modern NVIDIA GPUs (A100, RTX 30XX+) support TensorFloat-32
- Faster than FP32 (float32)
- Nearly as accurate as FP32
- ~2x speedup on matmul operations
- Transparent to the code (no changes needed)

```python
use_torch_compile = False
```

**Torch compile:** PyTorch 2.0+ feature that optimizes models
- Compiles model into faster code
- Significant speedup (1.5-2x)
- Can cause issues during development/debugging
- Disabled by default for stability

---

## Part 2: Trainer Class

The `Trainer` class encapsulates all training logic. Let's walk through each method.

### 2.1 Initialization (lines 24-55)

```python
class Trainer:
    def __init__(
            self, 
            model, 
            optimizer, 
            train_loader, 
            val_loader, 
            token_encoder, 
            eval_freq, 
            grad_accum_steps, 
            ddp, 
            ddp_rank, 
            ddp_world_size, 
            device, 
            logpath
    ):
```

**Parameters:**

- **`model`**: The GPT-2 model (possibly wrapped in DDP)
- **`optimizer`**: AdamW optimizer (handles weight updates)
- **`train_loader` / `val_loader`**: DataLoaderLite instances
- **`token_encoder`**: tiktoken encoder for text generation
- **`eval_freq`**: How often to evaluate (every N steps)
- **`grad_accum_steps`**: Number of micro-batches before weight update
- **`ddp`**: Boolean—is distributed training enabled?
- **`ddp_rank`**: Which GPU/process this is (0 to N-1)
- **`ddp_world_size`**: Total number of GPUs/processes
- **`device`**: 'cuda:0', 'cuda:1', 'cpu', etc.
- **`logpath`**: Where to save training logs

**Key concepts:**

**Gradient accumulation**: Simulate larger batches without more memory
- Example: `grad_accum_steps=4` with `B=32`
- Effectively trains with batch size of 128
- Only 32 sequences in memory at once

**Master process**: `ddp_rank == 0`
- Only this process does logging, printing, saving
- Prevents duplicate outputs from all GPUs

---

### 2.2 Main Training Loop (lines 58-137)

This is the heart of the entire training script. Let's break it down:

```python
def train(self, max_steps, warmup_steps, max_lr, min_lr):
    for step in range(max_steps):
```

**Loop structure:**
- Runs for `max_steps` iterations
- Each step processes one batch (or multiple micro-batches with gradient accumulation)
- Example: 19,073 steps/epoch × 5 epochs = 95,365 steps

#### Step 1: Timing and setup (lines 66-67)

```python
t0 = time.time()
self.is_last_step = (step == max_steps - 1)
```

- Records start time to measure iteration speed
- Flags the last step for final evaluation/checkpointing

#### Step 2: Validation evaluation (lines 70-71)

```python
if step % self.eval_freq == 0 or self.is_last_step:
    self.evaluate_validation(step)
```

**When it runs:**
- Every `eval_freq` steps (e.g., every 250 steps)
- On the very last step
- Measures how well model performs on unseen data

**Why periodic evaluation?**
- Tracks if model is overfitting
- Decides when to stop training
- Validates that learning is happening

#### Step 3: HellaSwag evaluation (lines 74-75)

```python
if ((step > 0 and step % self.eval_freq == 0) or self.is_last_step) and (not use_torch_compile):
    self.evaluate_helloswag(step)
```

**HellaSwag:** Common-sense reasoning benchmark
- Model completes sentences with context
- Tests real-world understanding
- Only runs if `torch.compile` is disabled (compile doesn't work with this eval)

#### Step 4: Text generation (lines 78-79)

```python
if ((step > 0 and step % self.eval_freq == 0) or self.is_last_step) and (not use_torch_compile):
    self.generate_sequences(num_seq=5, max_tokens=32)
```

**Purpose:** Sanity check that model is learning
- Generates 5 sample sequences
- Each sequence has 32 tokens
- Useful for qualitatively assessing progress
- Example: Early training produces gibberish → Later produces coherent text

#### Step 5: Training mode setup (lines 82-84)

```python
self.model.train()    # sets model to train mode
self.optimizer.zero_grad()    # resets all gradients
batch_loss = 0.0
```

**`model.train()`:** Enables training-specific behaviors
- Dropout is active (randomly zeros out neurons)
- BatchNorm updates running statistics
- Opposite: `model.eval()` disables these

**`optimizer.zero_grad()`:** Clears previous gradients
- PyTorch accumulates gradients by default
- Must clear before each optimization step

**`batch_loss`:** Tracks total loss across micro-batches

#### Step 6: Gradient accumulation loop (lines 86-109)

This is where the actual training happens!

```python
for mini_step in range(self.grad_accum_steps):
    inp, tar = self.train_loader.next_batch()
    inp, tar = inp.to(self.device), tar.to(self.device)
```

**Why loop?** Gradient accumulation simulates larger batches:
- `grad_accum_steps=4` means 4 mini-batches
- Gradients accumulate without updating weights
- After 4th mini-batch, weights update with averaged gradients

**Data loading:**
- `inp`: Input tokens (shape: `[B, T]`)
- `tar`: Target tokens (shape: `[B, T]`)
- `.to(self.device)`: Moves data from CPU to GPU

##### Forward Pass (lines 90-93)

```python
with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
    logits, loss = self.model(inp, tar)
```

**`torch.autocast`:** Automatic mixed precision training
- Operations use `bfloat16` (16-bit float) instead of `float32`
- **Benefits:**
  - 2x less memory
  - 1.5-2x faster computation
  - Minimal accuracy loss
- **GPU support:** Works on A100, H100, V100 (modern GPUs)

**Model forward pass:**
- Input: `inp` (token IDs)
- Output: 
  - `logits`: Predictions for next token (shape: `[B, T, vocab_size]`)
  - `loss`: Cross-entropy loss (scalar)

##### Loss Scaling (lines 95-99)

```python
loss /= self.grad_accum_steps
batch_loss += loss.detach()
```

**Why divide loss?**

Without gradient accumulation:
- 1 batch → compute loss → backward → update weights

With gradient accumulation (4 steps):
- Batch 1 → loss₁ → backward (gradients accumulate)
- Batch 2 → loss₂ → backward (gradients accumulate)
- Batch 3 → loss₃ → backward (gradients accumulate)
- Batch 4 → loss₄ → backward (gradients accumulate)
- Update weights with total gradient

**Problem:** Gradients add up, so they'd be 4x larger!

**Solution:** Divide loss by 4, so gradients have correct scale

**`batch_loss += loss.detach()`:** Track total loss for logging
- `.detach()` prevents building computation graph (saves memory)

##### DDP Gradient Synchronization (lines 101-104)

```python
if self.ddp:
    # in the final mini_step, sync and avg all gradients across all processes
    self.model.require_backward_grad_sync = (mini_step == self.grad_accum_steps - 1)
```

**DDP behavior:**

**Without gradient accumulation:**
- Each `loss.backward()` syncs gradients across GPUs
- Synchronization is expensive (communication overhead)

**With gradient accumulation:**
- First 3 mini-steps: Each GPU computes gradients independently (no sync)
- Last mini-step: Sync and average gradients across all GPUs
- **Benefit:** 4x less communication overhead

##### Backward Pass (line 109)

```python
loss.backward()
```

**What happens:**
1. Computes gradients for all model parameters
2. Gradients accumulate (not overwritten)
3. If DDP and final mini-step: Synchronizes gradients across GPUs

#### Step 7: DDP Loss Reduction (lines 111-115)

```python
if self.ddp:
    dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
```

**Why needed?**

- `batch_loss` is just a Python variable (not part of model)
- Not automatically synchronized by DDP
- Each GPU has different `batch_loss` (computed on different data)

**`all_reduce` operation:**
- Averages `batch_loss` across all GPUs
- Result deposited on all GPUs
- Now all GPUs have the same global loss value

#### Step 8: Gradient Clipping (line 118)

```python
norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
```

**What is gradient clipping?**

During training, gradients can sometimes explode (become very large):
- Causes weight updates that are too large
- Leads to NaN (not a number) losses
- Training crashes or becomes unstable

**Solution: Gradient clipping**

1. Compute global norm: \( \text{norm} = \sqrt{\sum_i \|\text{grad}_i\|^2} \)
2. If norm > 1.0, scale all gradients down: \( \text{grad}_i = \text{grad}_i \times \frac{1.0}{\text{norm}} \)
3. Effectively caps the gradient magnitude at 1.0

**Return value:** `norm` is the gradient norm (useful for debugging)

#### Step 9: Learning Rate Scheduling (lines 121-124)

```python
lr = self.estimate_lr(step, warmup_steps, max_steps, max_lr, min_lr)
for param_group in self.optimizer.param_groups:
    param_group['lr'] = lr
```

**Learning rate schedule:** Changes learning rate during training
- High LR early: Makes quick progress
- Low LR later: Fine-tunes without overshooting

**Implementation:**
- Calls `estimate_lr()` method (explained later)
- Updates optimizer's learning rate
- Applied before `optimizer.step()`

#### Step 10: Weight Update (line 126)

```python
self.optimizer.step()
```

**This is where learning happens!**

Updates weights using accumulated gradients:

\[
\theta_{\text{new}} = \theta_{\text{old}} - \text{lr} \times \text{grad}
\]

(Actually more complex with Adam: uses momentum and adaptive learning rates)

#### Step 11: GPU Synchronization (lines 127-128)

```python
if self.device_type == 'cuda':
    torch.cuda.synchronize()
```

**Why needed?**

- GPU operations are **asynchronous**
- `optimizer.step()` returns before GPU finishes
- Next line (timing) would measure wrong duration

**`synchronize()`:** Waits for all GPU operations to complete
- Ensures accurate timing
- Only used for measurement (not needed for correctness)

#### Step 12: Performance Metrics (lines 130-132)

```python
dt = (time.time() - t0) * 1000.0    # in ms
tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps * self.ddp_world_size
tokens_per_sec = tokens_processed / dt
```

**Calculates:**
- **`dt`**: Time per training step (milliseconds)
- **`tokens_processed`**: Total tokens in this step
  - `B=32`, `T=1024`, `grad_accum_steps=4`, `ddp_world_size=4`
  - Total = 32 × 1024 × 4 × 4 = 524,288 tokens
- **`tokens_per_sec`**: Training throughput

**Why track throughput?**
- Measures training efficiency
- Helps optimize hyperparameters
- Useful for estimating training time

#### Step 13: Logging (lines 134-137)

```python
if self.master_process:
    print(f'step {step:4d} | loss: {batch_loss.item():.6f} | lr: {lr:.2e} | norm: {norm:.4f} | dt: {dt:.4f}ms | tok/sec: {tokens_per_sec:.4f}')
    with open(self.logpath, 'a') as f:
        f.write(f'{step} train {batch_loss.item():.6f}\n')
```

**Only master process logs:** Prevents duplicate outputs

**Example log output:**
```
step  250 | loss: 3.456789 | lr: 6.00e-04 | norm: 0.8234 | dt: 234.5678ms | tok/sec: 2234.5678
```

**Log file format:**
```
250 train 3.456789
500 train 3.234567
750 train 3.123456
```

---

### 2.3 Validation Evaluation (lines 140-172)

```python
def evaluate_validation(self, step):
    self.model.eval()    # sets model to eval mode
    self.val_loader.reset()
```

**`model.eval()`:** Disables training-specific behaviors
- Dropout is turned off
- BatchNorm uses running statistics (doesn't update them)
- Results in deterministic behavior

**`val_loader.reset()`:** Starts from beginning of validation data
- Ensures consistent evaluation across steps
- Without reset, would evaluate different parts each time

#### Validation Loop (lines 144-153)

```python
with torch.no_grad():
    val_loss_accum = 0.0
    val_steps = 20
    for _ in range(val_steps):
        inp, tar = self.val_loader.next_batch()
        inp, tar = inp.to(self.device), tar.to(self.device)
        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
            logits, loss = self.model(inp, tar)
        loss /= val_steps
        val_loss_accum += loss.detach()
```

**`torch.no_grad()`:** Disables gradient computation
- Saves memory (doesn't store intermediate activations)
- Faster computation
- Gradients not needed for evaluation

**Why 20 steps?**
- Full validation set is large (100M tokens)
- Evaluating all would be slow
- 20 batches (640K tokens) is representative sample

**Loss averaging:** `loss /= val_steps` ensures proper scaling

#### DDP Loss Reduction (lines 155-156)

```python
if self.ddp:
    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
```

Similar to training loss reduction:
- Each GPU evaluated different data
- Average losses across all GPUs
- Get global validation loss

#### Logging and Checkpointing (lines 157-172)

```python
if self.master_process:
    print(f'Val loss: {val_loss_accum.item():.4f}')
    with open(self.logpath, 'a') as f:
        f.write(f'{step} val {val_loss_accum.item():.4f}\n')

    if step > 0 and (step % 10000 == 0 or self.is_last_step):
        raw_model = self.model.module if self.ddp else self.model
        logdir = os.path.dirname(self.logpath)
        ckpt_path = os.path.join(logdir, f'model_{step:05d}.pt')
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
            'step': step,
            'val_loss': val_loss_accum.item()
        }
        torch.save(checkpoint, ckpt_path)
```

**Checkpointing strategy:**
- Every 10,000 steps
- On the last step
- Saves model, config, and training state

**Why `raw_model = self.model.module`?**
- DDP wraps the model in an extra layer
- `.module` accesses the underlying model
- Checkpoint should contain unwrapped model

**Checkpoint contents:**
- **`model`**: All model weights (state_dict)
- **`config`**: Model architecture (allows reconstruction)
- **`step`**: Training progress
- **`val_loss`**: Performance metric

---

### 2.4 HellaSwag Evaluation (lines 175-207)

```python
def evaluate_helloswag(self, step):
```

**HellaSwag:** Common-sense natural language inference benchmark

**Task format:**
- **Context:** "She holds a large styrofoam ball and a small knife."
- **Continuations (4 options):**
  1. "She starts cutting the ball with the knife." ✓ (correct)
  2. "She throws the ball at the camera."
  3. "She eats the ball hungrily."
  4. "She juggles five balls simultaneously."

**Model task:** Predict which continuation is most likely

#### Example Processing (lines 182-194)

```python
for i, example in enumerate(iterate_examples('val')):
    if i % self.ddp_world_size != self.ddp_rank:
        continue
```

**Distributed evaluation:**
- Each GPU processes different examples
- Example 0 → GPU 0
- Example 1 → GPU 1
- Example 2 → GPU 2
- Example 3 → GPU 3
- Example 4 → GPU 0 (wraps around)

**Load and process example:**

```python
_, tokens, mask, label = render_example(example)
tokens, mask = tokens.to(self.device), mask.to(self.device)
```

- **`tokens`:** Shape `[4, N]` (4 continuations, N tokens each)
- **`mask`:** Shape `[4, N]` (marks which tokens to consider)
- **`label`:** Correct answer (0, 1, 2, or 3)

**Model evaluation:**

```python
with torch.no_grad():
    with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
        logits, loss = self.model(tokens)
    pred_norm = get_most_likely_row(tokens, mask, logits)
```

- **`logits`:** Model predictions for each token
- **`get_most_likely_row`:** Computes which row (continuation) is most likely
- **`pred_norm`:** Model's prediction (0, 1, 2, or 3)

**Accuracy tracking:**

```python
n_total += 1
n_correct_norm += int(pred_norm == label)
```

#### Reduce Results (lines 196-202)

```python
if self.ddp:
    n_total = torch.tensor(n_total, device=self.device, dtype=torch.long)
    n_correct_norm = torch.tensor(n_correct_norm, device=self.device, dtype=torch.long)
    dist.all_reduce(n_total, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_correct_norm, op=dist.ReduceOp.SUM)
    n_total = n_total.item()
    n_correct_norm = n_correct_norm.item()
```

**Why SUM instead of AVG?**
- Counting total correct predictions
- Summing counts gives global count
- Averaging wouldn't make sense here

#### Log Results (lines 203-207)

```python
acc_norm = n_correct_norm / n_total
if self.master_process:
    print(f'HelloSwag accuracy: {n_correct_norm}/{n_total}={acc_norm:.4f}')
    with open(self.logpath, 'a') as f:
        f.write(f'{step} hellaswag {acc_norm:.4f}\n')
```

**Example output:**
```
HelloSwag accuracy: 234/1000=0.2340
```

**Interpretation:**
- Random guessing: 25% (1/4 chance)
- Untrained model: ~25%
- GPT-2 124M: ~30-35%
- GPT-3 175B: ~78%

---

### 2.5 Text Generation (lines 210-237)

```python
def generate_sequences(self, num_seq=4, max_tokens=32):
    self.model.eval()
```

**Purpose:** Qualitatively check model behavior
- See if model produces coherent text
- Fun way to monitor training progress

#### Setup (lines 212-219)

```python
tokens = self.token_encoder.encode("Hello, I am a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_seq, 1)
gen_tokens = tokens.to(self.device)
sample_rng = torch.Generator(device=self.device)
sample_rng.manual_seed(42 + self.ddp_rank)
```

**Prompt:** "Hello, I am a language model"
- Starting point for generation
- Model continues from this prompt

**Why repeat `num_seq` times?**
- Generates multiple sequences in parallel
- Example: `num_seq=5` → generates 5 sequences simultaneously

**Separate RNG (random number generator):**
- Training uses global RNG (deterministic)
- Generation uses separate RNG
- Prevents generation from affecting training reproducibility

**Different seed per GPU:**
- `42 + self.ddp_rank` ensures different generations per GPU
- Otherwise all GPUs would generate identical text

#### Generation Loop (lines 221-232)

```python
while gen_tokens.shape[-1] <= max_tokens:
    with torch.no_grad():
        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
            logits, loss = self.model(gen_tokens)
        logits = logits[:, -1, :]    # (num_seq, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
        next_tok = torch.gather(topk_indices, -1, ix)
        gen_tokens = torch.cat([gen_tokens, next_tok], dim=1)
```

**Autoregressive generation:** One token at a time

**Step 1: Model forward pass**
```python
logits, loss = self.model(gen_tokens)
logits = logits[:, -1, :]
```

- Input: Current sequence (e.g., "Hello, I am")
- Output: Predictions for next token
- `[:, -1, :]` selects only the last position

**Step 2: Convert to probabilities**
```python
probs = F.softmax(logits, dim=-1)
```

- `logits`: Raw scores (can be negative, unbounded)
- `probs`: Probabilities (0 to 1, sum to 1)

**Step 3: Top-k sampling**
```python
topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
```

- Instead of considering all 50,257 tokens
- Only consider top 50 most likely tokens
- Prevents sampling nonsense/rare tokens

**Step 4: Sample from top-k**
```python
ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
next_tok = torch.gather(topk_indices, -1, ix)
```

- `multinomial`: Weighted random sampling
- Higher probability tokens more likely to be chosen
- Introduces randomness (diversity in generated text)

**Step 5: Append to sequence**
```python
gen_tokens = torch.cat([gen_tokens, next_tok], dim=1)
```

- Adds sampled token to sequence
- Next iteration uses longer sequence
- Continues until `max_tokens` reached

#### Display Results (lines 234-237)

```python
for i in range(num_seq):
    tokens = gen_tokens[i, :max_tokens].tolist()
    gen_text = self.token_encoder.decode(tokens)
    print(f"> rank {self.ddp_rank} sample {i}: {gen_text}")
```

**Example output early in training:**
```
> rank 0 sample 0: Hello, I am a language model jf#@$kjlsd asdfjkl...
> rank 0 sample 1: Hello, I am a language model ][][[][ asdfasdf...
```

**Example output after training:**
```
> rank 0 sample 0: Hello, I am a language model trained to assist you with various tasks including writing, coding, and analysis.
> rank 0 sample 1: Hello, I am a language model designed by OpenAI to help answer questions and provide information.
```

---

### 2.6 Learning Rate Scheduler (lines 240-254)

```python
def estimate_lr(self, step, warmup_steps, max_steps, max_lr, min_lr):
```

**Learning rate schedule:** Cosine decay with warmup

#### Phase 1: Linear Warmup (lines 245-246)

```python
if step < warmup_steps:
    return max_lr * (step+1) / warmup_steps
```

**Why warmup?**

Starting with high learning rate can destabilize training:
- Model weights are randomly initialized
- Large updates can push into bad regions
- Loss might explode or NaN

**Solution: Start small, gradually increase**

Example with `warmup_steps=715`, `max_lr=6e-4`:
- Step 0: LR = 6e-4 × 1/715 = 8.4e-7
- Step 357: LR = 6e-4 × 358/715 = 3e-4
- Step 714: LR = 6e-4 × 715/715 = 6e-4

#### Phase 2: Cosine Decay (lines 248-254)

```python
if step > max_steps:
    return min_lr
decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
assert 0 <= decay_ratio <= 1
coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
return min_lr + coeff * (max_lr - min_lr)
```

**Cosine decay:**

\[
\text{LR} = \text{min\_lr} + \frac{1}{2}(\text{max\_lr} - \text{min\_lr}) \times (1 + \cos(\pi \times \text{ratio}))
\]

**Visualization:**

```
LR
^
|     _____ max_lr
|    /     \
|   /       \___
|  /            \___
| /                 \______ min_lr
+--------------------------------> step
  warmup  |    cosine decay
```

**Why cosine instead of linear?**
- Smooth decay (no sudden changes)
- Spends more time near max_lr (faster learning)
- Gradually slows down near end (fine-tuning)

**Example values:**

With `max_lr=6e-4`, `min_lr=6e-5`, `warmup_steps=715`, `max_steps=95365`:

| Step | Phase | LR |
|------|-------|-----|
| 0 | Warmup | 8.4e-7 |
| 715 | End warmup | 6e-4 |
| 50000 | Mid-training | ~4e-4 |
| 90000 | Late training | ~1e-4 |
| 95365 | End | 6e-5 |

---

## Part 3: Configuration and Setup

### 3.1 GPTConfig Dataclass (lines 257-264)

```python
@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = 50257
    num_layers: int = 12
    embd_size: int = 768
    num_heads: int = 12
```

**Model architecture parameters:**

- **`context_length`**: Maximum sequence length (1024 tokens)
- **`vocab_size`**: Number of unique tokens (~50K)
- **`num_layers`**: Transformer blocks (12 = GPT-2 small)
- **`embd_size`**: Embedding dimension (768 = GPT-2 small)
- **`num_heads`**: Attention heads per layer (12 = GPT-2 small)

**GPT-2 model sizes:**

| Model | Layers | Embd Size | Heads | Parameters |
|-------|--------|-----------|-------|------------|
| Small | 12 | 768 | 12 | 124M |
| Medium | 24 | 1024 | 16 | 350M |
| Large | 36 | 1280 | 20 | 774M |
| XL | 48 | 1600 | 25 | 1.5B |

---

### 3.2 Command-Line Arguments (lines 266-285)

```python
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter Configuration")
```

**Key hyperparameters:**

```python
parser.add_argument("--total_batch_size", type=int, default=524288)
```
- **Total tokens per optimization step**
- Default: 524,288 = 2^19 tokens
- Matches GPT-3 paper's batch size

```python
parser.add_argument("--mini_batch_size", type=int, default=32)
```
- **Sequences per forward pass**
- Larger = better GPU utilization but more memory
- Typical: 8-64 depending on GPU

```python
parser.add_argument("--max_lr", type=float, default=1e-3)
parser.add_argument("--min_lr", type=float, default=1e-3 * 0.1)
```
- **Learning rate range**
- Max LR: Peak learning rate (after warmup)
- Min LR: Final learning rate (after decay)

```python
parser.add_argument("--warmup_steps", type=int, default=715)
```
- **Warmup duration**
- Typically 2-5% of total steps

```python
parser.add_argument("--weight_decay", type=float, default=0.1)
```
- **L2 regularization strength**
- Prevents overfitting
- Standard value: 0.1

```python
parser.add_argument("--steps_per_epoch", type=int, default=19073)
```
- **Steps to process entire dataset once**
- Calculation: 10^10 tokens ÷ 524,288 tokens/step ≈ 19,073

---

## Part 4: Main Function

### 4.1 Setup and Initialization (lines 288-300)

```python
def main():
    args = get_args()
    
    # Print hyperparameters
    print("Hyperparameter Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
```

**Why print configuration?**
- Useful for debugging
- Confirms correct hyperparameters
- Included in logs for reproducibility

```python
os.makedirs(args.logdir, exist_ok=True)
logpath = os.path.join(args.logdir, 'log.txt')
with open(logpath, 'w') as f:
    pass
```

**Log file initialization:**
- Creates log directory if it doesn't exist
- Clears/creates empty log file
- Prevents appending to old logs

---

### 4.2 Distributed Training Setup (lines 302-329)

```python
ddp = int(os.environ.get('RANK', -1)) != -1
```

**How DDP is detected:**
- `torchrun` command sets `RANK` environment variable
- If `RANK` exists → DDP mode
- Otherwise → single GPU/CPU mode

#### DDP Configuration (lines 307-317)

```python
if ddp:
    assert torch.cuda.is_available(), f'use of DDP requires CUDA'
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
```

**Key environment variables:**

- **`RANK`**: Global process ID (0 to N-1)
  - Used for data partitioning
  
- **`LOCAL_RANK`**: Local GPU ID on current machine
  - Used for device assignment
  
- **`WORLD_SIZE`**: Total number of processes
  - Used for gradient averaging

**Example with 2 machines, 4 GPUs each:**

| Machine | GPU | LOCAL_RANK | RANK | Device |
|---------|-----|------------|------|--------|
| Node 0 | 0 | 0 | 0 | cuda:0 |
| Node 0 | 1 | 1 | 1 | cuda:1 |
| Node 0 | 2 | 2 | 2 | cuda:2 |
| Node 0 | 3 | 3 | 3 | cuda:3 |
| Node 1 | 0 | 0 | 4 | cuda:0 |
| Node 1 | 1 | 1 | 5 | cuda:1 |
| Node 1 | 2 | 2 | 6 | cuda:2 |
| Node 1 | 3 | 3 | 7 | cuda:3 |

**`nccl` backend:** NVIDIA Collective Communications Library
- Optimized for multi-GPU communication
- Fastest option for NVIDIA GPUs

#### Non-DDP Configuration (lines 319-329)

```python
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')
```

**Device selection priority:**
1. CUDA (NVIDIA GPU)
2. MPS (Apple M1/M2 GPU)
3. CPU (fallback)

---

### 4.3 Reproducibility (lines 333-338)

```python
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
```

**Why set seeds?**
- Makes training deterministic
- Ensures reproducibility across runs
- Useful for debugging

**What gets seeded:**
- NumPy random operations
- PyTorch CPU random operations
- PyTorch GPU random operations (all GPUs)

**Note:** Even with same seed, DDP results might differ slightly due to:
- Floating-point precision
- GPU operation ordering
- DDP communication timing

---

### 4.4 Batch Size Calculation (lines 340-344)

```python
assert args.total_batch_size % (args.mini_batch_size * args.context_length * ddp_world_size) == 0
grad_accum_steps = args.total_batch_size // (args.mini_batch_size * args.context_length * ddp_world_size)
```

**Formula:**

\[
\text{grad\_accum\_steps} = \frac{\text{total\_batch\_size}}{\text{mini\_batch\_size} \times \text{context\_length} \times \text{num\_gpus}}
\]

**Example:**
- `total_batch_size` = 524,288 tokens
- `mini_batch_size` = 32 sequences
- `context_length` = 1024 tokens/sequence
- `ddp_world_size` = 4 GPUs

\[
\text{grad\_accum\_steps} = \frac{524288}{32 \times 1024 \times 4} = \frac{524288}{131072} = 4
\]

**Interpretation:**
- Each GPU processes 32 sequences per forward pass
- 4 forward passes before weight update
- Effectively: 32 × 4 × 4 = 512 sequences per optimization step

---

### 4.5 DataLoader Creation (lines 347-348)

```python
train_loader = DataLoaderLite(B=args.mini_batch_size, T=args.context_length, 
                               process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=args.mini_batch_size, T=args.context_length, 
                             process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
```

**Creates separate loaders for:**
- **Training data:** 99 shards (~9.9B tokens)
- **Validation data:** 1 shard (~100M tokens)

---

### 4.6 Model Creation (lines 350-364)

```python
gpt_config = GPTConfig(vocab_size=50304,
                       context_length=args.context_length, 
                       num_layers=args.num_layers, 
                       num_heads=args.num_heads, 
                       embd_size=args.embd_size)
model = GPT(config=gpt_config)
model.to(device)
```

**Why `vocab_size=50304` instead of 50257?**

- Actual vocab size: 50,257
- Rounded up to 50,304
- **Reason:** 50,304 = 64 × 786 (highly divisible)
- Better GPU performance (memory alignment)
- Embeddings are matrix: `[vocab_size, embd_size]`
- GPU operations faster when dimensions are multiples of 64/128

**Torch Compile (lines 361-364):**

```python
if use_torch_compile:
    model = torch.compile(model)
```

- Optimizes model for faster execution
- Typical speedup: 1.5-2x
- Disabled by default (compilation takes time)

**DDP Wrapper (lines 366-370):**

```python
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```

- Wraps model for distributed training
- Handles gradient synchronization
- Transparent to training code

---

### 4.7 Optimizer Setup (lines 372-374)

```python
raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, 
                                            lr=args.max_lr, 
                                            device_type=device_type, 
                                            master_process=master_process)
token_encoder = tiktoken.get_encoding('gpt2')
```

**Why `raw_model`?**
- DDP wraps model in extra layer
- Optimizer configuration in model class
- Need unwrapped model to access method

**Optimizer:** Typically AdamW
- Adaptive learning rates per parameter
- Weight decay for regularization
- Default optimizer for transformers

---

### 4.8 Training Execution (lines 376-382)

```python
start_time = time.time()
trainer = Trainer(model, optimizer, train_loader, val_loader, token_encoder, 
                  args.eval_freq, grad_accum_steps, 
                  ddp, ddp_rank, ddp_world_size, device, logpath)

max_steps = args.steps_per_epoch * args.num_epochs
trainer.train(max_steps, args.warmup_steps, args.max_lr, args.min_lr)
```

**Training duration:**
- `max_steps` = 19,073 × 5 = 95,365 steps
- At ~200ms per step: ~5.3 hours on 4x A100 GPUs

**Timing (lines 384-385):**

```python
dt = (time.time() - start_time) / (60*60)
print(f"Total training time: {dt:.4f}hr")
```

---

### 4.9 Cleanup (lines 387-388)

```python
if ddp:
    dist.destroy_process_group()
```

**Why needed?**
- Cleanly shuts down distributed training
- Releases communication resources
- Prevents hanging processes

---

## Training Workflow Summary

Here's the complete flow when you run `python train.py`:

1. **Parse arguments** → Get hyperparameters
2. **Setup DDP** → Initialize multi-GPU communication
3. **Set seeds** → Ensure reproducibility
4. **Create data loaders** → Setup data pipeline
5. **Create model** → Initialize GPT-2 architecture
6. **Create optimizer** → Setup AdamW
7. **Create trainer** → Wrap everything together
8. **Training loop** (for each step):
   - Evaluate validation loss (periodically)
   - Evaluate HellaSwag (periodically)
   - Generate text samples (periodically)
   - Load batch from DataLoader
   - Forward pass (with gradient accumulation)
   - Backward pass (compute gradients)
   - Clip gradients
   - Update learning rate
   - Optimizer step (update weights)
   - Log metrics
9. **Save checkpoint** (periodically)
10. **Cleanup** → Destroy process group

---

## Running the Training

### Single GPU

```bash
python train.py --mini_batch_size 32 --context_length 1024
```

### Multi-GPU (4 GPUs)

```bash
torchrun --standalone --nproc_per_node=4 train.py --mini_batch_size 32 --context_length 1024
```

**`torchrun` arguments:**
- `--standalone`: Single machine (not distributed across nodes)
- `--nproc_per_node=4`: Use 4 GPUs

### Multi-Node (2 nodes, 4 GPUs each)

**Node 0:**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=29500 train.py
```

**Node 1:**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=29500 train.py
```

---

## Performance Tuning

### Memory Optimization

1. **Reduce batch size:** Decrease `mini_batch_size`
2. **Reduce context length:** Use `--context_length 512` instead of 1024
3. **Enable gradient checkpointing:** (requires model.py modification)
4. **Use bfloat16:** Already enabled with `autocast`

### Speed Optimization

1. **Increase batch size:** Maximize `mini_batch_size` without OOM
2. **Enable torch.compile:** Set `use_torch_compile = True`
3. **Use more GPUs:** Better scaling with 4-8 GPUs
4. **Optimize data loading:** Ensure data is on fast SSD

### Example Configurations

**Budget GPU (RTX 3090, 24GB):**
```bash
python train.py --mini_batch_size 16 --context_length 512
```

**High-end GPU (A100 80GB):**
```bash
python train.py --mini_batch_size 64 --context_length 1024
```

**4x A100 40GB:**
```bash
torchrun --standalone --nproc_per_node=4 train.py --mini_batch_size 32 --context_length 1024
```

---

## Monitoring Training

### Key Metrics

**Training loss:** Should steadily decrease
- Initial: ~10-11 (random model)
- After 1 epoch: ~3.5-4.0
- After 5 epochs: ~2.8-3.2
- GPT-2 paper: ~2.5-3.0

**Validation loss:** Should track training loss
- If much higher: overfitting
- If similar: good generalization

**HellaSwag accuracy:** Measures reasoning ability
- Random: 25%
- Untrained: ~25%
- After training: 30-35% (GPT-2 124M)

**Learning rate:** Should follow schedule
- Warmup: Linear increase
- Training: Cosine decay
- End: Reaches min_lr

**Gradient norm:** Should be stable
- Typical: 0.5-2.0
- If >5: Gradients might be exploding
- Clipping ensures max = 1.0

---

## Common Issues

### Issue: CUDA Out of Memory

**Solutions:**
1. Reduce `mini_batch_size`
2. Reduce `context_length`
3. Reduce `grad_accum_steps`
4. Use fewer layers/smaller model

### Issue: NaN Loss

**Causes:**
- Learning rate too high
- Gradient explosion
- Numerical instability

**Solutions:**
1. Reduce `max_lr`
2. Increase `warmup_steps`
3. Check gradient norm (should be clipped at 1.0)

### Issue: Slow Training

**Solutions:**
1. Enable `torch.compile`
2. Increase `mini_batch_size`
3. Use faster GPU
4. Verify data is on SSD (not HDD)

### Issue: Loss Not Decreasing

**Causes:**
- Learning rate too low
- Bug in code
- Data issues

**Solutions:**
1. Increase `max_lr`
2. Verify data loads correctly
3. Check model forward pass

---

## Key Takeaways

1. **Gradient accumulation** simulates large batches without memory cost
2. **Distributed training** splits work across GPUs for faster training
3. **Mixed precision** (bfloat16) saves memory and speeds up training
4. **Learning rate schedule** is crucial for good performance
5. **Gradient clipping** prevents training instability
6. **Periodic evaluation** helps monitor training progress
7. **Checkpointing** enables resuming training and model deployment

---

## Next Steps

After understanding the training script:
1. **Run training**: Start with small model to verify everything works
2. **Monitor logs**: Watch metrics to ensure training is healthy
3. **Experiment**: Try different hyperparameters
4. **Evaluate**: Load checkpoints and test model performance
5. **Deploy**: Use trained model for inference

Happy training! 🚀

