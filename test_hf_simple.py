#!/usr/bin/env python3
"""
Simple test script matching the user's original test code, with fixes.
Uses GPT2Tokenizer (not AutoTokenizer) and supports bfloat16.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import json

# Model path
model_path = "./hf_model"

print("="*80)
print("Simple Hugging Face Model Test")
print("="*80)
print(f"Model path: {model_path}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print()

# Check if model exists
if not Path(model_path).exists():
    print(f"✗ Error: Model path '{model_path}' does not exist!")
    exit(1)

# Load tokenizer (use GPT2Tokenizer, not AutoTokenizer)
print("Loading tokenizer...")
try:
    # Use GPT-2 tokenizer from Hugging Face
    # The model directory doesn't have tokenizer files, so we use the standard GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"✗ Failed to load tokenizer: {e}")
    exit(1)

# Load model
print("\nLoading model...")
try:
    # Determine dtype from config
    with open(Path(model_path) / "config.json", 'r') as f:
        config_data = json.load(f)
        config_dtype = config_data.get('torch_dtype', 'bfloat16')
    
    if config_dtype == 'bfloat16':
        model_dtype = torch.bfloat16
    elif config_dtype == 'float16':
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    
    print(f"  Loading with dtype: {config_dtype}")
    
    model = GPT2LMHeadModel.from_pretrained(
        model_path,
        dtype=model_dtype,
        trust_remote_code=False
    )
    model.eval()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Move to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Test 1: Plain text (original test)
print("\n" + "="*80)
print("TEST 1: Plain Text Generation (Original Test)")
print("="*80)
prompt = "the secret to baking a really good cake is"
print(f"Prompt: '{prompt}'")

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=100,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nResult:\n{result}")

# Test 2: Instruction format (what the model was trained on)
print("\n" + "="*80)
print("TEST 2: Instruction Format (Recommended)")
print("="*80)
instruction = "What is Python?"
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
print(f"Instruction: {instruction}")

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=len(inputs['input_ids'][0]) + 150,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

full_text = tokenizer.decode(output[0], skip_special_tokens=False)
if "### Response:\n" in full_text:
    response = full_text.split("### Response:\n")[-1].split("### Instruction:")[0].strip()
else:
    response = full_text[len(prompt):].strip()

print(f"\nResponse:\n{response}")

print("\n" + "="*80)
print("✓ Tests completed!")
print("="*80)
print("\nNote: The model was fine-tuned on instruction format, so Test 2")
print("      should produce better results than Test 1 (plain text).")
