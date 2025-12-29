# GPT-2 Inference Guide

## 🎯 Quick Start

```bash
# Activate environment
cd /home/kecso/Documents/workspace/training/gpt-2/gpt2-from-scratch
source .venv/bin/activate

# Run inference with default prompt
python src/inference.py

# Run with custom prompt
python src/inference.py --prompt "Your text here" --num_seq 5 --max_tokens 50
```

---

## 📋 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompt` | str | `"Hello, I am a language model,"` | Input text to continue from |
| `--num_seq` | int | `5` | Number of sequences to generate |
| `--max_tokens` | int | `50` | Maximum tokens per sequence |

---

## 📝 Examples

### Generate Educational Text
```bash
python src/inference.py --prompt "In machine learning" --num_seq 3 --max_tokens 60
```

### Generate Programming Content
```bash
python src/inference.py --prompt "def calculate" --num_seq 3 --max_tokens 80
```

### Short Completions
```bash
python src/inference.py --prompt "The best way to learn" --num_seq 5 --max_tokens 30
```

---

## 🔧 What Was Fixed (2025-11-23)

### Problem
```python
# inference.py line 83 was trying to access:
model = GPT(config=checkpoint['config'])
# But checkpoint didn't have 'config' key!
```

### Solution 1: Update Checkpoint Saving
Added model config to checkpoint in `train_improved.py`:

```python
checkpoint = {
    # ... other fields ...
    'config': model.config,  # ← NEW: Save model config for inference
    # ...
}
```

### Solution 2: Update Inference Script
Made `inference.py` backward-compatible:

```python
# Handles 3 scenarios:
if 'config' in checkpoint:
    # New checkpoints - use saved config
    config = checkpoint['config']
elif 'args' in checkpoint:
    # Old checkpoints - reconstruct from training args
    config = GPTConfig(
        vocab_size=50304,
        context_length=ckpt_args.get('context_length', 1024),
        num_layers=ckpt_args.get('num_layers', 12),
        num_heads=ckpt_args.get('num_heads', 12),
        embd_size=ckpt_args.get('embd_size', 768)
    )
else:
    # Fallback - use default config
    config = GPTConfig()
```

---

## 💡 Key Points

1. **Works with existing checkpoints**: Old checkpoints without `config` field will reconstruct config from training args
2. **Future checkpoints will be better**: New checkpoints saved after this fix will include the config directly
3. **No data loss**: Your trained models are fully usable for inference
4. **Automatic config detection**: The script detects which checkpoint format you have

---

## 🎨 Model Hyperparameters

Your trained model uses:
- **Vocabulary Size**: 50,304 tokens
- **Context Length**: 1,024 tokens
- **Number of Layers**: 12
- **Embedding Size**: 768
- **Number of Heads**: 12

---

## 🚀 Performance Tips

1. **Batch Generation**: The script generates multiple sequences in parallel
2. **GPU Acceleration**: Automatically uses CUDA if available
3. **BFloat16**: Uses bfloat16 precision for faster inference
4. **Top-k Sampling**: Uses top-50 sampling for diversity

---

## 📊 Output Format

Each generation shows:
```
> sample 0: [generated text]
> sample 1: [generated text]
> sample 2: [generated text]
...
```

---

## 🔍 Troubleshooting

### Issue: "KeyError: 'config'"
**Solution**: Already fixed! The script now handles old checkpoints automatically.

### Issue: Model generates poor text
**Possible causes**:
- Model hasn't trained enough yet
- Try different prompts
- Adjust `max_tokens` or `num_seq`

### Issue: CUDA out of memory
**Solution**: Reduce `num_seq` (number of parallel generations)

---

## 📁 Checkpoint Locations

The inference script looks for checkpoints at:
- `./checkpoints/best_model.pt` (default)

You can modify `model_path` in `inference.py` to use other checkpoints:
- `./checkpoints/latest.pt`
- `./checkpoints/rolling_step_XXXXXX.pt`
- `./checkpoints/emergency_step_XXXXXX_TIMESTAMP.pt`

---

## 🎓 How It Works

1. **Load Checkpoint**: Loads model weights and config
2. **Initialize Model**: Creates GPT model with correct architecture
3. **Load Weights**: Applies trained weights to model
4. **Generate**: Uses top-k sampling with temperature for text generation
5. **Decode**: Converts token IDs back to text

---

## 🌟 Next Steps

- Experiment with different prompts
- Try adjusting `max_tokens` for longer/shorter completions
- Use different checkpoints to see training progress
- Integrate inference into your applications

Enjoy your trained GPT-2 model! 🎉

