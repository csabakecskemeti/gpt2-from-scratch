# 🔄 Convert Checkpoint to Hugging Face Format

## Quick Start

Convert your instruction fine-tuned checkpoint to Hugging Face safetensors format:

```bash
# Install safetensors if needed
pip install safetensors

# Convert checkpoint (default: bfloat16, recommended)
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt --output_dir hf_model --dtype bfloat16

# Or use the shell script
./convert_to_hf.sh
```

---

## Usage

### Basic Conversion (BF16 - Recommended)

```bash
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt
```

This creates `hf_model/` with:
- `model.safetensors` - Model weights in safetensors format
- `config.json` - Hugging Face model configuration
- `tokenizer_config.json` - Tokenizer configuration
- `README.md` - Model documentation

### Custom Output Directory

```bash
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt --output_dir my_hf_model --dtype bfloat16
```

### Convert Specific Checkpoint

```bash
python src/convert_to_safetensor.py checkpoints_instruct/step_002000.pt --output_dir hf_model_step2000
```

### Different Precision Options

```bash
# BF16 (default, recommended for inference)
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt --dtype bfloat16

# Float32 (full precision)
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt --dtype float32

# Float16 (smallest file size)
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt --dtype float16
```

---

## Using with Hugging Face Transformers

### Load the Model

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json

# Load your fine-tuned model
# Note: Use GPT2Tokenizer, not AutoTokenizer (model dir doesn't have tokenizer files)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Determine dtype from config
with open("hf_model/config.json", 'r') as f:
    config_data = json.load(f)
    config_dtype = config_data.get('torch_dtype', 'bfloat16')

if config_dtype == 'bfloat16':
    model_dtype = torch.bfloat16
elif config_dtype == 'float16':
    model_dtype = torch.float16
else:
    model_dtype = torch.float32

model = GPT2LMHeadModel.from_pretrained("hf_model", dtype=model_dtype)
model.eval()

# Generate with instruction format
prompt = "### Instruction:\nWhat is Python?\n\n### Response:\n"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(
    inputs,
    max_length=200,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Upload to Hugging Face Hub (Optional)

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="hf_model",
    repo_id="your-username/gpt2-instruct",
    repo_type="model"
)
```

---

## What Gets Converted

✅ **Model weights** → `model.safetensors`
- All model parameters in safetensors format
- Compatible with HF Transformers

✅ **Model config** → `config.json`
- Architecture: GPT-2
- All hyperparameters (layers, heads, dimensions, etc.)

✅ **Tokenizer config** → `tokenizer_config.json`
- Basic tokenizer settings
- Note: You'll still use `GPT2Tokenizer.from_pretrained("gpt2")` for actual tokenization

✅ **Documentation** → `README.md`
- Model description
- Usage examples

---

## Requirements

```bash
pip install safetensors transformers
```

---

## Notes

1. **Tokenizer**: Use `GPT2Tokenizer.from_pretrained("gpt2")`, NOT `AutoTokenizer` (model directory doesn't have tokenizer files)
2. **Vocab Size**: Your model has vocab_size=50304 (vs standard GPT-2's 50257)
3. **Format**: The instruction format (`### Instruction:\n...\n\n### Response:\n`) should be preserved
4. **Compatibility**: Works with all HF Transformers features (generation, fine-tuning, etc.)
5. **Precision**: BF16 is recommended for inference (default) - same dynamic range as float32, half the memory
6. **Dtype Parameter**: Use `dtype` not `torch_dtype` when loading (deprecated in newer transformers versions)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'safetensors'"
```bash
pip install safetensors
```

### "KeyError" or weight mismatch
- Make sure you're using the correct checkpoint
- Check that vocab_size matches (50304)

### Model generates poorly after conversion
- Ensure you're using the instruction format in prompts
- Check temperature and sampling parameters

---

## Example: Full Pipeline

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("hf_model")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# Format instruction
instruction = "What is machine learning?"
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

# Tokenize
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=len(inputs[0]) + 150,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode
full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = full_text.split("### Response:\n")[-1]
print(response)
```

---

**Ready to convert?** Run:
```bash
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt
```

**Or test the conversion:**
```bash
python test_hf_conversion.py
```


