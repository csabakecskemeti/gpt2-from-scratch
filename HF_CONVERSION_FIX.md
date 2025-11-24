# Hugging Face Conversion Fix

## Problem

When converting the PyTorch checkpoint to Hugging Face safetensors format, the model failed to load with this error:

```
RuntimeError: Error(s) in loading state_dict for Conv1D:
    size mismatch for weight: copying a param with shape torch.Size([2304, 768]) 
    from checkpoint, the shape in current model is torch.Size([768, 2304]).
```

## Root Cause

The issue is a **weight transpose mismatch** between our custom model and Hugging Face GPT-2:

1. **Our Model**: Uses `nn.Linear` layers which store weights as `(out_features, in_features)`
   - Example: `c_attn.weight` shape is `(2304, 768)` for a layer that projects 768 -> 2304

2. **Hugging Face GPT-2**: Uses `Conv1D` layers (which are actually linear layers with different naming) that store weights as `(in_features, out_features)` - **transposed!**
   - Example: `c_attn.weight` shape should be `(768, 2304)` for the same layer

## Solution

The conversion script (`src/convert_to_safetensor.py`) now transposes the following weights when converting:

- `transformer.h.{i}.attn.c_attn.weight` - Attention QKV projection
- `transformer.h.{i}.attn.c_proj.weight` - Attention output projection  
- `transformer.h.{i}.mlp.c_fc.weight` - MLP input projection
- `transformer.h.{i}.mlp.c_proj.weight` - MLP output projection

These correspond to the Linear layers in our model that Hugging Face stores as Conv1D.

## Changes Made

1. **`src/convert_to_safetensor.py`** (new script):
   - Added logic to detect and transpose Conv1D weights
   - Transposes weights from `(out, in)` to `(in, out)` format
   - Preserves all other weights (embeddings, layer norms, biases) unchanged
   - Supports bfloat16, float16, and float32 precision
   - Better parameter counting and verification

2. **Test scripts**:
   - Updated `test_hf_inference.py`, `test_hf_simple.py`, and `test_hf_conversion.py` to use `dtype` instead of deprecated `torch_dtype`
   - Fixed config attribute access (`n_layer` instead of `num_layers`)
   - All use `GPT2Tokenizer.from_pretrained("gpt2")` instead of `AutoTokenizer`

## How to Re-convert

If you already converted a model and it's not working, re-run the conversion:

```bash
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt --output_dir hf_model --dtype bfloat16
```

The new conversion will:
1. Load the checkpoint
2. Transpose the Conv1D weights correctly
3. Save as safetensors with proper shapes

## Verification

After re-converting, test with:

```bash
# Simple test
python test_hf_simple.py

# Comprehensive test
python test_hf_conversion.py

# Or detailed inference test
python test_hf_inference.py
```

The model should now load successfully with Hugging Face Transformers!

## Technical Details

This transpose requirement comes from how GPT-2 was originally implemented. Hugging Face's GPT-2 uses a `Conv1D` class (which is actually a linear layer) that stores weights transposed compared to PyTorch's standard `nn.Linear`. This is a historical artifact from the original OpenAI implementation.

When loading from Hugging Face to our model (see `model.py` line 156-166), we transpose back. When converting from our model to Hugging Face, we need to transpose forward.

