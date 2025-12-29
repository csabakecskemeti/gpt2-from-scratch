"""
Interactive chat interface for instruction-tuned GPT-2 model.
"""

import torch
import tiktoken
from model import GPT, GPTConfig


def load_model(checkpoint_path, device='cuda'):
    """Load fine-tuned model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Fallback: reconstruct from args
        args = checkpoint.get('args', {})
        config = GPTConfig(
            vocab_size=50304,
            context_length=args.get('context_length', 1024),
            num_layers=args.get('num_layers', 12),
            num_heads=args.get('num_heads', 12),
            embd_size=args.get('embd_size', 768)
        )
    
    # Create and load model
    model = GPT(config=config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    return model, config


def generate_response(model, tokenizer, instruction, max_new_tokens=150, 
                     temperature=0.7, top_k=50, device='cuda'):
    """Generate response to an instruction"""
    
    # Format with template
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    # Tokenize
    tokens = tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for _ in range(max_new_tokens):
                # Forward pass
                logits, _ = model(tokens)
                logits = logits[:, -1, :]  # Get last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                    logits = logits_filtered
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # Stop if we hit end of text or newline patterns that suggest end
                if next_token.item() == tokenizer.eot_token:
                    break
    
    # Decode
    generated_text = tokenizer.decode(tokens[0].tolist())
    
    # Extract just the response part
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1]
    else:
        response = generated_text
    
    # Clean up
    response = response.replace('<|endoftext|>', '').strip()
    
    return response


def interactive_chat(checkpoint_path, device='cuda'):
    """Run interactive chat loop"""
    
    # Load model
    model, config = load_model(checkpoint_path, device)
    
    # Load tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    
    # Print instructions
    print("="*80)
    print("Interactive Chat with Instruction-Tuned GPT-2")
    print("="*80)
    print("Type your instructions or questions below.")
    print("Commands:")
    print("  'exit' or 'quit' - Exit the chat")
    print("  'clear' - Clear conversation history")
    print("="*80)
    print()
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n\033[1;34mYou:\033[0m ")
            
            # Check for commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'clear':
                print("\n" * 50)  # Clear screen
                print("="*80)
                print("Conversation cleared")
                print("="*80)
                continue
            
            if not user_input.strip():
                continue
            
            # Generate response
            print("\n\033[1;32mAssistant:\033[0m ", end='', flush=True)
            response = generate_response(
                model, tokenizer, user_input,
                max_new_tokens=200,
                temperature=0.7,
                top_k=50,
                device=device
            )
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit or continue chatting.")
            continue
        except Exception as e:
            print(f"\n\033[1;31mError:\033[0m {e}")
            continue


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive chat with instruction-tuned GPT-2")
    parser.add_argument('--checkpoint', type=str, default='checkpoints_instruct/latest.pt',
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--max_tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling parameter')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run chat
    interactive_chat(args.checkpoint, args.device)


if __name__ == "__main__":
    main()

