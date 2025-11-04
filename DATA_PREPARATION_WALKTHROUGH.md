# Data Preparation Walkthrough

This document provides a comprehensive walkthrough of the `prepare_dataset.py` script, which downloads and tokenizes the FineWeb-Edu dataset for GPT-2 training.

## Overview

The script downloads the FineWeb-Edu dataset (a 10 billion token educational text corpus) and converts it into tokenized shards that can be efficiently loaded during training.

## Key Concepts

### What is Tokenization?
Tokenization converts raw text into numerical tokens that the model can process. For example:
- Text: "Hello, world!"
- Tokens: [15496, 11, 995, 0]

### What are Shards?
Instead of one giant file, the data is split into manageable chunks called "shards". Each shard contains exactly 100 million tokens. This makes it easier to load data during training without consuming all available RAM.

---

## Code Walkthrough

### 1. Setup & Configuration (lines 10-25)

```python
script_dir = os.path.dirname(__file__)
local_dir = os.path.join(script_dir, "../data/edu_fineweb10B")
remote_name = "sample-10BT"
shard_size = int(1e8)    # 100M tokens per shard, total 100 shards = 10B gpt2 tokens
```

**Configuration:**
- `local_dir`: Where to save the processed data
- `remote_name`: Which dataset variant to download (10BT = 10 billion tokens)
- `shard_size`: 100 million tokens per shard

**Why 100M tokens per shard?**
- Large enough to be efficient
- Small enough to fit in memory during training
- With 10B total tokens: 10B ÷ 100M = 100 shards

---

### 2. Download Dataset (line 28)

```python
fw = load_dataset('HuggingFaceFW/fineweb-edu', name=remote_name, split='train')
```

Downloads the FineWeb-Edu dataset from HuggingFace. This is where your `HF_TOKEN` environment variable is used for authentication.

**FineWeb-Edu Dataset:**
- High-quality educational web content
- Filtered for educational value
- ~10 billion tokens in the sample variant

---

### 3. Tokenizer Setup (lines 30-32)

```python
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']
```

**tiktoken:** OpenAI's fast tokenizer library
- Converts text to tokens using GPT-2's BPE (Byte-Pair Encoding) vocabulary
- Vocabulary size: ~50,257 tokens

**`<|endoftext|>` Token:**
- Special token that marks document boundaries
- Crucial for the model to distinguish between separate documents
- Without it, the model might think one document continues into another

---

### 4. Tokenization Function (lines 34-41)

```python
def tokenize(doc):
    """ tokenizes a single document and returns a np array of uint16 tokens """
    tokens = [eot]    # special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (tokens_np >= 0).all() and (tokens_np < 2**16).all(), 'token dict too large for uint16'
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16
```

**Step-by-step process:**

1. **Start with `<|endoftext|>`**: Every document begins with this delimiter
2. **Encode text**: Convert the document's text into token IDs
3. **Convert to NumPy array**: For efficient numerical processing
4. **Verify range**: Ensure all token IDs fit in uint16 (0 to 65,535)
5. **Cast to uint16**: Store as 16-bit integers

**Why uint16?**
- GPT-2's vocabulary has ~50k tokens, which fits easily in uint16 (max 65,536)
- Uses half the memory of uint32 (2 bytes vs 4 bytes)
- For 10B tokens: saves ~20GB of disk space!

---

### 5. Parallel Processing Setup (lines 44-50)

```python
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_idx = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
```

**Multiprocessing:**
- Uses half your CPU cores for parallel tokenization
- Much faster than processing documents sequentially
- Example: 16-core CPU → 8 processes → ~8x speedup

**Buffer allocation:**
- `all_tokens_np`: A pre-allocated array for exactly 100M tokens
- Think of it as a bucket that we fill up
- `token_count`: Tracks how full the bucket currently is

---

### 6. Main Processing Loop - Simple Case (lines 52-60)

```python
for tokens in pool.imap(tokenize, fw, chunksize=16):
    # check if there is enough space in current shard for new tokens
    if token_count + len(tokens) < shard_size:
        # simply append tokens to current shard
        all_tokens_np[token_count : token_count + len(tokens)] = tokens
        token_count += len(tokens)
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'shard {shard_idx}')
        progress_bar.update(len(tokens))
```

**When tokens fit in current shard:**
1. Check if there's enough space
2. Copy tokens into the buffer
3. Update the counter
4. Update progress bar

**`pool.imap` parameters:**
- `tokenize`: The function to apply to each document
- `fw`: The dataset iterator
- `chunksize=16`: Process 16 documents at a time per worker (balances overhead vs efficiency)

---

### 7. Main Processing Loop - Overflow Case (lines 61-73)

```python
else:
    # write current shard and start a new one
    split = 'val' if shard_idx == 0 else 'train'
    filepath = os.path.join(DATA_CACHE_DIR, f'edufineweb_{split}_{shard_idx:06d}')
    # split the document into whatever fits in this shard, remainder goes to next one
    remainder = shard_size - token_count
    progress_bar.update(remainder)
    all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
    np.save(filepath, all_tokens_np)
    shard_idx += 1
    progress_bar = None
    all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
    token_count = len(tokens) - remainder
```

**When tokens don't fit:**

1. **Determine split type**: First shard = validation, rest = training
2. **Calculate remainder**: How much space is left in current shard?
3. **Fill current shard**: Add tokens until it's exactly 100M
4. **Save to disk**: Write as `.npy` file (NumPy's efficient binary format)
5. **Start new shard**: Put leftover tokens at the beginning of next shard
6. **Update counter**: Track how many tokens are in the new shard

**Example:**
- Current shard has 99,950,000 tokens
- New document has 100,000 tokens
- Fill with first 50,000 tokens → save shard
- Put remaining 50,000 tokens in new shard

**File naming:**
- `edufineweb_val_000000.npy`: First shard (validation)
- `edufineweb_train_000001.npy`: Second shard (training)
- `edufineweb_train_000002.npy`: Third shard (training)
- etc.

---

### 8. Final Shard Handling (lines 75-78)

```python
if token_count != 0:
    split = 'val' if shard_idx == 0 else 'train'
    filepath = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_idx:06d}")
    np.save(filepath, all_tokens_np[:token_count])
```

**Handles the last partial shard:**
- After processing all documents, there will likely be leftover tokens
- This code saves whatever remains, even if it's less than 100M tokens
- Uses `[:token_count]` to only save the filled portion of the buffer

---

## Key Design Decisions

### 1. Streaming Processing
**Instead of:** Load all 10B tokens into RAM → process → save  
**Actually:** Process one document at a time, save shards incrementally  
**Benefit:** Uses constant memory regardless of dataset size

### 2. Efficient Storage
**uint16 instead of int32/int64:**
- Saves 50-75% disk space
- Faster to load during training
- 10B tokens: ~20GB vs 40-80GB

### 3. Parallel Processing
**Multiprocessing with `pool.imap`:**
- Tokenizes multiple documents simultaneously
- Utilizes multiple CPU cores
- Typical speedup: 5-8x on modern machines

### 4. Train/Val Split
**First shard = validation, rest = training:**
- Simple and effective splitting strategy
- ~1% validation data (100M out of 10B tokens)
- Ensures validation set is from same distribution

---

## Expected Output

After running the script, you'll have:

```
data/edu_fineweb10B/
├── edufineweb_val_000000.npy      # 100M tokens (validation)
├── edufineweb_train_000001.npy    # 100M tokens (training)
├── edufineweb_train_000002.npy    # 100M tokens (training)
├── ...
└── edufineweb_train_000099.npy    # ~100M tokens (training, last shard may be smaller)
```

**Total:** ~100 files, ~20GB of data

---

## Performance Metrics

**Typical processing time:**
- Dataset size: 10 billion tokens
- Processing speed: 50,000-100,000 tokens/second (depends on CPU)
- Total time: 30-60 minutes on a modern workstation

**Resource usage:**
- CPU: 50% utilization (half your cores)
- RAM: ~2-4 GB (constant, thanks to streaming)
- Disk I/O: Sequential writes (very efficient)

---

## Common Issues & Solutions

### Issue: Permission Denied Error
**Symptom:** `PermissionError: [Errno 13] Permission denied`  
**Cause:** HuggingFace cache directory owned by root  
**Solution:** 
```bash
sudo chown -R $USER:$USER ~/.cache/huggingface/
```

### Issue: Out of Memory
**Symptom:** Process killed or `MemoryError`  
**Cause:** Too many parallel processes  
**Solution:** Reduce `nprocs` manually in the script

### Issue: Slow Download
**Symptom:** Dataset download takes very long  
**Cause:** Network speed or HuggingFace server load  
**Solution:** Be patient; download is cached and only happens once

---

## Understanding the Math

### Total Tokens Calculation
- FineWeb-Edu sample-10BT ≈ 10 billion tokens
- Shard size = 100 million tokens
- Number of shards = 10B ÷ 100M = 100 shards

### Memory Efficiency
- Token ID range: 0 to 50,256
- Storage options:
  - uint8: Can't fit (max 255)
  - uint16: Perfect fit (max 65,535) ✓
  - uint32: Wastes space (max 4 billion)
- Memory savings: 10B tokens × 2 bytes = 20GB vs 40GB (uint32)

### Processing Pipeline
```
Raw Text → Tokenizer → Token IDs → uint16 Array → Disk (as .npy)
```

---

## Next Steps

After data preparation completes:
1. Verify output files exist in `data/edu_fineweb10B/`
2. Check file sizes (each should be ~200MB for 100M uint16 tokens)
3. Proceed to training with `train.py`

The training script will automatically load these shards using the `DataLoaderLite` class.


