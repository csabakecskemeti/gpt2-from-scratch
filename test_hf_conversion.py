#!/usr/bin/env python3
"""
Test script for the converted Hugging Face safetensors model.
This script tests the model loading and basic inference.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import sys
import json


def test_model_conversion(model_path="./hf_model"):
    """
    Test the converted model with basic inference.
    
    Args:
        model_path: Path to the converted model directory
    """
    print("="*80)
    print("Testing Hugging Face Model Conversion")
    print("="*80)
    print(f"Model path: {model_path}")
    print()
    
    # Check if model exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"✗ Error: Model path '{model_path}' does not exist!")
        return False
    
    required_files = ["model.safetensors", "config.json"]
    missing = [f for f in required_files if not (model_path_obj / f).exists()]
    if missing:
        print(f"✗ Error: Missing required files: {missing}")
        return False
    
    print("✓ Model directory found with required files")
    print()
    
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
        print(f"  Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False
    
    print()
    
    # Load model
    print("Loading model...")
    try:
        # Determine dtype from config
        with open(model_path_obj / "config.json", 'r') as f:
            config_data = json.load(f)
            config_dtype = config_data.get('torch_dtype', 'float32')
        
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
        print(f"  Model type: {type(model)}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Config: {model.config.n_layer} layers, {model.config.n_embd} dim, {model.config.n_head} heads")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 1: Plain text generation
    print("="*80)
    print("TEST 1: Plain Text Generation")
    print("="*80)
    prompt = "the secret to baking a really good cake is"
    print(f"Prompt: '{prompt}'")
    print()
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
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
        print("✓ Generation successful")
        print(f"\nResult:\n{result}")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 2: Instruction format (what the model was trained on)
    print("="*80)
    print("TEST 2: Instruction Format Generation")
    print("="*80)
    instruction = "What is Python?"
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    print(f"Instruction: {instruction}")
    print()
    
    try:
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
        
        # Extract response
        if "### Response:\n" in full_text:
            response = full_text.split("### Response:\n")[-1]
            response = response.split("### Instruction:")[0].strip()
        else:
            response = full_text[len(prompt):].strip()
        
        print("✓ Generation successful")
        print(f"\nResponse:\n{response}")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("="*80)
    print("✓ All tests passed!")
    print("="*80)
    return True


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./hf_model"
    success = test_model_conversion(model_path)
    sys.exit(0 if success else 1)

