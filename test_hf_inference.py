#!/usr/bin/env python3
"""
Test script for the converted Hugging Face safetensors model.
Tests both plain text generation and instruction-following.
"""

import torch
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
from pathlib import Path
import sys

def test_model_loading(model_path):
    """Test loading the model"""
    print("="*80)
    print("TEST 1: Model Loading")
    print("="*80)
    
    try:
        # Try AutoModelForCausalLM
        print("\n[1a] Trying AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            dtype=torch.float32,
            trust_remote_code=False
        )
        model.eval()
        print("✓ AutoModelForCausalLM loaded successfully")
        print(f"  Model type: {type(model)}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model, "auto"
    except Exception as e:
        print(f"✗ AutoModelForCausalLM failed: {e}")
        
        try:
            # Try GPT2LMHeadModel
            print("\n[1b] Trying GPT2LMHeadModel...")
            model = GPT2LMHeadModel.from_pretrained(
                model_path,
                dtype=torch.float32,
                trust_remote_code=False
            )
            model.eval()
            print("✓ GPT2LMHeadModel loaded successfully")
            print(f"  Model type: {type(model)}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            return model, "gpt2"
        except Exception as e2:
            print(f"✗ GPT2LMHeadModel failed: {e2}")
            raise


def test_tokenizer_loading(model_path):
    """Test loading the tokenizer"""
    print("\n" + "="*80)
    print("TEST 2: Tokenizer Loading")
    print("="*80)
    
    try:
        # Try AutoTokenizer from model path
        print("\n[2a] Trying AutoTokenizer.from_pretrained(model_path)...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ AutoTokenizer loaded from model path")
        return tokenizer, "auto_model"
    except Exception as e:
        print(f"✗ AutoTokenizer from model path failed: {e}")
        print("  (This is expected - model directory doesn't have tokenizer files)")
        
        try:
            # Use GPT-2 tokenizer (standard)
            print("\n[2b] Using GPT2Tokenizer.from_pretrained('gpt2')...")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
            # Set pad token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("✓ GPT2Tokenizer loaded successfully")
            print(f"  Vocab size: {len(tokenizer)}")
            print(f"  Model vocab size (from config): 50304")
            print(f"  Note: Model has custom vocab_size=50304, but GPT-2 tokenizer should work")
            return tokenizer, "gpt2"
        except Exception as e2:
            print(f"✗ GPT2Tokenizer failed: {e2}")
            raise


def test_plain_text_generation(model, tokenizer, device='cuda'):
    """Test plain text generation (not instruction format)"""
    print("\n" + "="*80)
    print("TEST 3: Plain Text Generation")
    print("="*80)
    
    prompt = "the secret to baking a really good cake is"
    
    print(f"\nPrompt: '{prompt}'")
    print("-"*80)
    
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = model.to(device)
        
        print(f"Input tokens: {inputs['input_ids'].shape}")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n✓ Generation successful")
        print(f"\nGenerated text:")
        print(f"{generated_text}")
        print("-"*80)
        return True
        
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def format_instruction(instruction, input_text=None):
    """Format instruction in the training format"""
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def test_instruction_generation(model, tokenizer, device='cuda'):
    """Test instruction-following generation"""
    print("\n" + "="*80)
    print("TEST 4: Instruction-Following Generation")
    print("="*80)
    
    test_cases = [
        ("What is Python?", None),
        ("Explain machine learning in simple terms.", None),
        ("Summarize the following text", "Machine learning is a subset of artificial intelligence that enables computers to learn from data."),
    ]
    
    model = model.to(device)
    
    for i, (instruction, input_text) in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"Instruction: {instruction}")
        if input_text:
            print(f"Input: {input_text}")
        print("-"*80)
        
        try:
            # Format prompt
            prompt = format_instruction(instruction, input_text)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=len(inputs['input_ids'][0]) + 150,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract response
            if "### Response:\n" in full_text:
                response = full_text.split("### Response:\n")[-1]
                response = response.split("### Instruction:")[0].strip()
            else:
                response = full_text[len(prompt):].strip()
            
            print(f"✓ Generation successful")
            print(f"\nResponse:")
            print(f"{response}")
            
        except Exception as e:
            print(f"✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()


def test_batch_generation(model, tokenizer, device='cuda'):
    """Test batch generation"""
    print("\n" + "="*80)
    print("TEST 5: Batch Generation")
    print("="*80)
    
    instructions = [
        "What is Python?",
        "Explain neural networks.",
        "What is the difference between AI and ML?"
    ]
    
    print(f"\nProcessing {len(instructions)} instructions in batch...")
    
    try:
        # Format prompts
        prompts = [format_instruction(inst) for inst in instructions]
        
        # Tokenize batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = model.to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 100,
                temperature=0.7,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and display
        print("\n✓ Batch generation successful")
        for i, (inst, output) in enumerate(zip(instructions, outputs), 1):
            full_text = tokenizer.decode(output, skip_special_tokens=False)
            if "### Response:\n" in full_text:
                response = full_text.split("### Response:\n")[-1].split("### Instruction:")[0].strip()
            else:
                response = full_text[len(prompts[i-1]):].strip()
            
            print(f"\n[{i}] Instruction: {inst}")
            print(f"    Response: {response[:200]}...")
            
    except Exception as e:
        print(f"✗ Batch generation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Setup
    model_path = "./hf_model"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("Hugging Face Model Inference Test")
    print("="*80)
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check if model exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"\n✗ Error: Model path '{model_path}' does not exist!")
        print(f"  Make sure you've converted the checkpoint:")
        print(f"  python src/convert_to_safetensor.py checkpoints_instruct/latest.pt --output_dir hf_model --dtype bfloat16")
        return
    
    # Check required files
    required_files = ["model.safetensors", "config.json"]
    missing = [f for f in required_files if not (model_path_obj / f).exists()]
    if missing:
        print(f"\n✗ Error: Missing required files: {missing}")
        return
    
    print("\n✓ Model directory found with required files")
    
    try:
        # Test 1: Load model
        model, model_type = test_model_loading(model_path)
        
        # Test 2: Load tokenizer
        tokenizer, tokenizer_type = test_tokenizer_loading(model_path)
        
        # Test 3: Plain text generation
        test_plain_text_generation(model, tokenizer, device)
        
        # Test 4: Instruction generation
        test_instruction_generation(model, tokenizer, device)
        
        # Test 5: Batch generation
        test_batch_generation(model, tokenizer, device)
        
        print("\n" + "="*80)
        print("✓ All tests completed!")
        print("="*80)
        print(f"\nModel type used: {model_type}")
        print(f"Tokenizer type used: {tokenizer_type}")
        print("\nNote: If instruction generation works better than plain text,")
        print("      that's expected - the model was fine-tuned on instruction format.")
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ Test suite failed with error:")
        print("="*80)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

