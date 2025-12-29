# DataLoader Walkthrough

This document provides a comprehensive walkthrough of the `dataloader.py` script, which implements an efficient data loading system for training GPT-2 on the FineWeb-Edu dataset.

## Overview

The `DataLoaderLite` class is a lightweight, custom data loader designed specifically for:
- **Distributed training**: Works seamlessly with multiple GPUs/processes
- **Efficient memory usage**: Loads one shard at a time instead of entire dataset
- **Continuous streaming**: Automatically cycles through shards for multiple epochs

## Key Concepts

### What is a DataLoader?
A DataLoader handles:
1. **Loading data** from disk into memory
2. **Batching** data into properly sized chunks for training
3. **Managing state** across multiple training iterations

### Why Custom DataLoader?
Instead of using PyTorch's built-in DataLoader, we implement our own because:
- **Simplicity**: Our data is pre-tokenized and stored as NumPy arrays
- **Efficiency**: Directly maps to the shard structure created by `prepare_dataset.py`
- **Distributed support**: Built-in handling for multi-GPU/multi-process training

---

## Code Walkthrough

### 1. Initialization (lines 8-27)

```python
class DataLoaderLite:
    """ A simple dataloader for FineWebEdu-10B dataset """

    def __init__(self, B, T, process_rank, num_processes, split='train'):
        super().__init__()
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
```

**Parameters explained:**

- **`B` (Batch size)**: Number of sequences processed simultaneously
  - Example: `B=32` means 32 sequences per batch
  - Larger `B` = better GPU utilization but more memory
  
- **`T` (Context length)**: Number of tokens per sequence
  - Example: `T=1024` means each sequence has 1024 tokens
  - Matches the model's maximum context window
  
- **`process_rank`**: Which GPU/process this is (0, 1, 2, ...)
  - In single-GPU training: always 0
  - In multi-GPU training: 0 to N-1 (where N = number of GPUs)
  
- **`num_processes`**: Total number of GPUs/processes
  - Single GPU: 1
  - 4 GPUs: 4
  
- **`split`**: Either 'train' or 'val' (validation)

**Key idea:** Each process loads the **same shard** but reads **different portions** of it, ensuring no duplicate data across GPUs.

---

### 2. Loading Shard Files (lines 18-26)

```python
# get the shard filenames
data_root = os.path.join(script_dir, "../data/edu_fineweb10B")
shard_filenames = os.listdir(data_root)
shard_filenames = sorted([filename for filename in shard_filenames if split in filename])
self.shard_filepaths = [os.path.join(data_root, filename) for filename in shard_filenames]
assert len(self.shard_filepaths) > 0, f'no shards found for split {split}'
master_process = process_rank == 0
if master_process:
    print(f'found {len(self.shard_filepaths)} shards for split {split}')
```

**What happens here:**

1. **Locate data directory**: Points to where shards were saved by `prepare_dataset.py`
2. **List all files**: Gets all filenames in the directory
3. **Filter by split**: 
   - If `split='train'`: keeps files with 'train' in name
   - If `split='val'`: keeps files with 'val' in name
4. **Sort files**: Ensures consistent ordering across all processes
5. **Store full paths**: Converts filenames to absolute paths
6. **Validation**: Crashes if no shards found (prevents silent failures)

**Example output for training:**
```
found 99 shards for split train
```

**Example file structure:**
```
data/edu_fineweb10B/
├── edufineweb_val_000000.npy       # val split
├── edufineweb_train_000001.npy     # train split
├── edufineweb_train_000002.npy     # train split
└── ...
```

---

### 3. Loading Tokens from Disk (lines 29-31)

```python
def load_tokens(self, filepath):
    tokens = torch.tensor(np.load(filepath).astype(np.int32), dtype=torch.long)
    return tokens
```

**Step-by-step process:**

1. **`np.load(filepath)`**: Loads the `.npy` file from disk
   - Returns a NumPy array of uint16 values
   - Shape: `(100_000_000,)` for full shards
   
2. **`.astype(np.int32)`**: Converts uint16 → int32
   - Why? PyTorch doesn't support uint16 well
   - Temporarily uses more memory during conversion
   
3. **`torch.tensor(..., dtype=torch.long)`**: Creates PyTorch tensor
   - `torch.long` = int64 in PyTorch
   - Final format compatible with embedding layers

**Memory note:** A shard of 100M tokens uses:
- On disk: 200 MB (uint16, 2 bytes per token)
- In RAM: 800 MB (int64, 8 bytes per token)

---

### 4. Resetting State (lines 33-37)

```python
def reset(self):
    # state, init at shard 0
    self.curr_shard = 0
    self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
    self.curr_pos = self.B * self.T * self.process_rank
```

**Called when:**
- Initialization (in `__init__`, line 27)
- Beginning of validation (to ensure fresh start)
- Manually if you want to restart from first shard

**State tracking:**

- **`self.curr_shard = 0`**: Start from the first shard
- **`self.tokens = ...`**: Load the first shard into memory
- **`self.curr_pos = B * T * process_rank`**: Set starting position

**Multi-GPU positioning example:**

Suppose `B=32`, `T=1024`, and 4 GPUs:
- **GPU 0**: starts at position 0
- **GPU 1**: starts at position 32,768 (32 × 1024 × 1)
- **GPU 2**: starts at position 65,536 (32 × 1024 × 2)
- **GPU 3**: starts at position 98,304 (32 × 1024 × 3)

Each GPU reads a different chunk, ensuring no data overlap!

---

### 5. Getting Next Batch (lines 39-49)

```python
def next_batch(self):
    B, T = self.B, self.T
    batch = self.tokens[self.curr_pos : self.curr_pos + B*T + 1]
    x_batch = batch[:-1].view(B, T)
    y_batch = batch[1:].view(B, T)
    self.curr_pos += B * T * self.num_processes
    if self.curr_pos + (B * T + 1) > len(self.tokens):
        self.curr_shard = (self.curr_shard + 1) % len(self.shard_filepaths)
        self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
        self.curr_pos = self.B * self.T * self.process_rank
    return x_batch, y_batch
```

This is the **core function** that gets called repeatedly during training. Let's break it down:

#### Step 1: Extract batch slice (line 41)

```python
batch = self.tokens[self.curr_pos : self.curr_pos + B*T + 1]
```

**Why `B*T + 1`?**
- We need `B × T` tokens for input
- Plus 1 extra token for the target (shifted by one position)
- Example: If `B=32` and `T=1024`, we extract 32,769 tokens

#### Step 2: Create input and target sequences (lines 42-43)

```python
x_batch = batch[:-1].view(B, T)
y_batch = batch[1:].view(B, T)
```

**This implements the language modeling objective:**

Given a sequence of tokens `[t₀, t₁, t₂, t₃, t₄]`:
- **Input (`x_batch`)**: `[t₀, t₁, t₂, t₃]`
- **Target (`y_batch`)**: `[t₁, t₂, t₃, t₄]`

The model learns to predict the **next token**.

**Visual example:**

```
Original: [15496, 11, 262, 995, 318, 1049]  (length = 6)

x_batch:  [15496, 11, 262, 995, 318]        (first 5 tokens)
y_batch:  [11, 262, 995, 318, 1049]         (last 5 tokens)
```

The model predicts:
- Given 15496, predict 11
- Given 11, predict 262
- Given 262, predict 995
- etc.

**Shape transformation:**

Before `.view(B, T)`: flat array of `B × T` tokens  
After `.view(B, T)`: 2D tensor of shape `(B, T)`

Example with `B=2`, `T=3`:
```
Before: [10, 20, 30, 40, 50, 60]
After:  [[10, 20, 30],
         [40, 50, 60]]
```

#### Step 3: Update position for next call (line 44)

```python
self.curr_pos += B * T * self.num_processes
```

**Why multiply by `num_processes`?**

In distributed training, multiple processes read from the **same shard simultaneously**:

Example with 4 GPUs, `B=32`, `T=1024`:
- **GPU 0** reads tokens 0-32,767 (first call)
- **GPU 1** reads tokens 32,768-65,535
- **GPU 2** reads tokens 65,536-98,303
- **GPU 3** reads tokens 98,304-131,071

On the **next call**:
- **GPU 0** reads tokens 131,072-163,839 (skips what others read)
- **GPU 1** reads tokens 163,840-196,607
- etc.

Each GPU advances by `B × T × 4` = 131,072 tokens.

#### Step 4: Handle shard boundaries (lines 45-48)

```python
if self.curr_pos + (B * T + 1) > len(self.tokens):
    self.curr_shard = (self.curr_shard + 1) % len(self.shard_filepaths)
    self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
    self.curr_pos = self.B * self.T * self.process_rank
```

**What triggers this?**

When the next batch would go beyond the current shard's length.

**What happens:**

1. **Move to next shard**: `self.curr_shard + 1`
2. **Wrap around**: The `% len(...)` ensures we cycle back to shard 0 after the last shard
   - Enables multi-epoch training without manual intervention
3. **Load new shard**: Reads next shard from disk
4. **Reset position**: Each process returns to its designated starting position

**Example scenario:**

Suppose a shard has 100M tokens and `curr_pos = 99,999,000`:
- Next batch needs tokens `99,999,000` to `100,032,768` (33,769 tokens)
- But shard only has 1,000 tokens left!
- **Solution**: Move to next shard and start fresh

---

## Multi-GPU Training Details

### Data Parallel Training Strategy

**Goal:** Train faster by splitting work across multiple GPUs

**Approach:**
1. Each GPU has an **identical copy** of the model
2. Each GPU processes **different data**
3. Gradients are **averaged** across GPUs after backward pass

### How DataLoader Enables This

The DataLoader ensures each GPU sees different data through:

**Initial positioning:**
```python
self.curr_pos = self.B * self.T * self.process_rank
```

**Position advancement:**
```python
self.curr_pos += B * T * self.num_processes
```

### Example: 4 GPUs, B=8, T=1024

**First call to `next_batch()`:**
- GPU 0: reads positions 0 to 8,191
- GPU 1: reads positions 8,192 to 16,383
- GPU 2: reads positions 16,384 to 24,575
- GPU 3: reads positions 24,576 to 32,767

**Second call to `next_batch()`:**
- All GPUs skip ahead by `8 × 1024 × 4 = 32,768`
- GPU 0: reads positions 32,768 to 40,959
- GPU 1: reads positions 40,960 to 49,151
- GPU 2: reads positions 49,152 to 57,343
- GPU 3: reads positions 57,344 to 65,535

This pattern continues throughout training!

---

## Design Decisions Explained

### 1. Why Load One Shard at a Time?

**Alternative approach:** Load all 10B tokens into RAM

**Problems:**
- Would require ~80 GB RAM (10B × 8 bytes)
- Most systems can't handle this
- Slow startup time

**Our approach:** Load one shard (~800 MB)
- **Memory efficient**: Only 0.8 GB per process
- **Fast startup**: Immediately start training
- **Automatic cycling**: Seamlessly moves to next shard

### 2. Why Pre-compute Batch Boundaries?

The loader doesn't split documents or add padding—it treats all shards as one continuous stream of tokens.

**Advantages:**
- **Simplicity**: No complex document boundary handling
- **Efficiency**: No wasted computation on padding tokens
- **Maximum data utilization**: Every token is used for training

**Potential concern:** A document might be split across batches

**Why it's okay:**
- The `<|endoftext|>` tokens mark document boundaries
- The model learns to recognize these delimiters
- Effectively teaches the model document segmentation

### 3. Why Cycle Through Shards?

```python
self.curr_shard = (self.curr_shard + 1) % len(self.shard_filepaths)
```

The `%` (modulo) operator makes the iterator **infinite**:
- After the last shard, automatically returns to shard 0
- Enables training for multiple epochs without manual reset
- Training script controls when to stop (via `max_steps`)

---

## Usage in Training

Here's how `DataLoaderLite` is typically used in `train.py`:

```python
# Initialize loaders
train_loader = DataLoaderLite(
    B=32,                    # mini-batch size
    T=1024,                  # context length
    process_rank=ddp_rank,   # GPU id
    num_processes=ddp_world_size,  # total GPUs
    split='train'
)

val_loader = DataLoaderLite(
    B=32,
    T=1024,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split='val'
)

# Training loop
for step in range(max_steps):
    # Get batch
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    # Forward pass
    logits, loss = model(x, y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

**Note:** The loader handles all the complexity of:
- Multi-GPU coordination
- Shard loading
- Position tracking
- Batch creation

The training loop just calls `next_batch()` repeatedly!

---

## Performance Characteristics

### Memory Usage

**Per process:**
- Loaded shard: ~800 MB (100M tokens × 8 bytes)
- Current batch: ~0.25 MB (32 × 1024 × 8 bytes)
- **Total**: ~800 MB per GPU

### Loading Speed

- **First shard load**: ~1-2 seconds (reading from disk)
- **Subsequent loads**: Faster if OS caches data
- **Batch extraction**: ~0.1-0.5 ms (in-memory array slicing)

**Loading is rarely a bottleneck:**
- Shard loads every ~3,000 batches (100M ÷ 32K)
- GPU training time per batch: ~100-500 ms
- Loading time is <1% of total training time

### Throughput

With `B=32`, `T=1024`:
- Each `next_batch()` call returns 32,768 tokens
- At 10ms per call: ~3.3M tokens/second per GPU
- With 4 GPUs: ~13M tokens/second

---

## Common Issues & Solutions

### Issue: Out of Memory When Loading Shard

**Symptom:** `RuntimeError: CUDA out of memory`

**Cause:** 
- Shard (~800 MB) + model + activations exceed GPU memory

**Solutions:**
1. **Reduce batch size** (`B`): Smaller batches use less memory
2. **Reduce context length** (`T`): Shorter sequences use less memory
3. **Use gradient accumulation**: Simulate larger batches without more memory

### Issue: Slow Training Start

**Symptom:** Long pause before first training step

**Cause:** Loading first shard from slow storage (HDD)

**Solution:**
- Move data to SSD (much faster reads)
- Pre-load shard in background thread (advanced optimization)

### Issue: Validation Loss Not Improving

**Symptom:** Training loss decreases but validation loss doesn't

**Cause (possibly):** Validation set too small (only 1 shard = 100M tokens)

**Solution:**
- Not a DataLoader issue—consider using more validation shards
- Modify `prepare_dataset.py` to create more val shards

---

## Key Takeaways

1. **Simple but effective**: ~50 lines of code handle complex distributed data loading
2. **Memory efficient**: Loads one shard at a time, uses <1 GB per GPU
3. **Distributed-ready**: Built-in multi-GPU support with no data overlap
4. **Infinite iterator**: Automatically cycles through shards for multi-epoch training
5. **Zero-copy operations**: Uses tensor slicing and views (no unnecessary copying)

---

## Comparison with PyTorch DataLoader

| Feature | DataLoaderLite | PyTorch DataLoader |
|---------|----------------|-------------------|
| **Complexity** | ~50 lines | Hundreds of lines |
| **Shuffling** | No (sequential) | Yes (configurable) |
| **Multiprocessing** | Manual (DDP) | Built-in workers |
| **Batching** | Custom logic | Automatic collation |
| **Memory** | One shard only | Configurable |
| **Flexibility** | Specific to our task | General-purpose |

**Why we use custom:**
- Our data is already pre-processed and well-structured
- Don't need shuffling (language modeling works with sequential data)
- Simpler code = easier to understand and debug
- Direct control over distributed behavior

---

## Next Steps

After understanding the DataLoader:
1. **Read `model.py`**: Understand how the model processes batches
2. **Read `train.py`**: See how DataLoader integrates into training loop
3. **Experiment**: Try modifying `B` and `T` to see memory/speed trade-offs

The DataLoader is a critical but often overlooked component. Understanding it helps debug training issues and optimize performance!

