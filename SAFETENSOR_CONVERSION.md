# Safetensor Conversion Guide

## New Conversion Script

A brand new conversion script has been created based on the approach from [pytorch-to-safetensor-converter](https://github.com/Silver267/pytorch-to-safetensor-converter).

### Script: `src/convert_to_safetensor.py` (Working Version)

This is the working conversion script that properly converts PyTorch checkpoints to Hugging Face safetensors format with:

1. **Proper Conv1D Weight Transposition**: Transposes weights for `c_attn`, `c_proj`, `c_fc` layers to match Hugging Face's Conv1D format
2. **Shared Weight Handling**: Clones shared weights (`lm_head.weight` and `transformer.wte.weight`) to make them independent
3. **Contiguous Tensors**: Ensures all tensors are contiguous for efficient loading
4. **Precision Options**: Supports `float32` (full precision), `float16` (half precision), and `bfloat16` (recommended for inference)
5. **Complete HF Setup**: Creates `config.json`, `tokenizer_config.json`, and `README.md`

## Usage

### Basic Conversion (BF16 - Recommended)

```bash
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt --output_dir hf_model
```

BF16 is the default and recommended for inference (same dynamic range as float32, half the memory).

### Conversion with Float32 (Full Precision)

```bash
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt --output_dir hf_model --dtype float32
```

### Conversion with Float16 (Smallest File Size)

```bash
python src/convert_to_safetensor.py checkpoints_instruct/latest.pt --output_dir hf_model --dtype float16
```

### Using the Shell Script

```bash
./convert_to_hf.sh                              # Convert latest checkpoint
./convert_to_hf.sh step_002000                  # Convert specific step
./convert_to_hf.sh checkpoints_instruct/latest.pt my_model  # Custom paths
```

## Command Line Options

```
positional arguments:
  checkpoint            Path to checkpoint file (e.g., checkpoints_instruct/latest.pt)

optional arguments:
  --output_dir OUTPUT_DIR
                        Output directory for HF-compatible model (default: hf_model)
  --model_name MODEL_NAME
                        Model name for README (default: gpt2-instruct)
  --vocab_size VOCAB_SIZE
                        Vocabulary size (default: 50304)
  --dtype {float32,float16,bfloat16}
                        Output dtype: float32 (full precision), float16 (half precision), or bfloat16 (recommended for inference) (default: bfloat16)
```

## What Gets Converted

The script handles:

1. **Weight Transposition**: 
   - `transformer.h.{i}.attn.c_attn.weight` (QKV projection)
   - `transformer.h.{i}.attn.c_proj.weight` (Attention output)
   - `transformer.h.{i}.mlp.c_fc.weight` (MLP input)
   - `transformer.h.{i}.mlp.c_proj.weight` (MLP output)

2. **Shared Weights**: 
   - Clones `lm_head.weight` and `transformer.wte.weight` to make them independent

3. **Other Weights**: 
   - Embeddings, layer norms, biases remain unchanged (just made contiguous)

## Output Files

The conversion creates:

- `model.safetensors` - Model weights in safetensors format
- `config.json` - Hugging Face model configuration
- `tokenizer_config.json` - Tokenizer configuration
- `README.md` - Model documentation

## Testing

After conversion, test with:

```bash
# Simple test (matches user's original test code)
python test_hf_simple.py

# Comprehensive test suite
python test_hf_conversion.py

# Or the detailed inference test
python test_hf_inference.py
```

**Note**: All test scripts use `GPT2Tokenizer.from_pretrained("gpt2")` instead of `AutoTokenizer` because the model directory doesn't include tokenizer files.

## Differences from Old Script

The new script (`convert_to_safetensor.py`) is a complete rewrite that:

1. Follows the pattern from the reference implementation
2. Properly handles tensor contiguity (important for safetensors)
3. Has better error handling and progress reporting
4. Supports precision conversion (float16/float32)
5. More robust weight handling

## Notes

- **Memory Requirements**: Ensure you have at least 2x the checkpoint size in available RAM
- **BF16 Recommended**: BF16 is the default and recommended for inference - same dynamic range as float32 but half the memory
- **Float16 Warning**: Converting to float16 may cause NaN values in rare cases. Use bfloat16 or float32 if you encounter issues.
- **Tokenizer**: The model directory doesn't include tokenizer files. Use `GPT2Tokenizer.from_pretrained("gpt2")` separately (not `AutoTokenizer`).
- **Parameter Count**: The converted model may show a different parameter count when loaded (due to vocab size differences), but all weights are correctly converted.

## References

- Based on: [pytorch-to-safetensor-converter](https://github.com/Silver267/pytorch-to-safetensor-converter)
- Safetensors library: [safetensors](https://github.com/huggingface/safetensors)

