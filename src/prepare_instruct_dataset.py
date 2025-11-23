"""
Prepare Alpaca GPT-4 dataset for instruction fine-tuning.
Downloads the dataset, formats it with instruction template, tokenizes, and saves as shards.
"""

import os
import json
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def format_instruction(example):
    """
    Format a single example with instruction template.
    Alpaca format: {instruction, input, output}
    """
    instruction = example['instruction'].strip()
    input_text = example.get('input', '').strip()
    output = example['output'].strip()
    
    # Build formatted text
    if input_text:
        # If there's additional input/context
        formatted = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        # Just instruction and response
        formatted = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return formatted


def main():
    # Configuration
    dataset_name = "vicgalle/alpaca-gpt4"
    output_dir = "data_instruct"
    shard_size = 100  # Examples per shard
    
    print("="*80)
    print("Instruction Dataset Preparation")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directories
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)
    
    # Load dataset
    print("Downloading dataset from Hugging Face...")
    dataset = load_dataset(dataset_name)
    
    # Alpaca only has 'train' split, we'll split it ourselves
    full_data = dataset['train']
    total_examples = len(full_data)
    
    print(f"Total examples: {total_examples:,}")
    
    # Create train/val split (95/5)
    indices = np.random.RandomState(42).permutation(total_examples)
    val_size = int(0.05 * total_examples)
    val_indices = set(indices[:val_size])
    
    print(f"Train examples: {total_examples - val_size:,}")
    print(f"Val examples: {val_size:,}")
    print()
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Process and save shards
    train_buffer = []
    val_buffer = []
    train_shard_idx = 0
    val_shard_idx = 0
    
    print("Processing examples...")
    for idx, example in enumerate(tqdm(full_data, desc="Formatting")):
        # Format with template
        formatted_text = format_instruction(example)
        
        # Tokenize
        tokens = tokenizer.encode(formatted_text, allowed_special={'<|endoftext|>'})
        tokens.append(tokenizer.eot_token)  # Add end of text token
        
        # Add to appropriate buffer
        if idx in val_indices:
            val_buffer.extend(tokens)
            
            # Save validation shard if buffer is full
            if len(val_buffer) >= shard_size * 512:  # Approximate tokens per shard
                filename = f"{output_dir}/val/val_shard_{val_shard_idx:04d}.npy"
                np.save(filename, np.array(val_buffer, dtype=np.uint16))
                val_buffer = []
                val_shard_idx += 1
        else:
            train_buffer.extend(tokens)
            
            # Save training shard if buffer is full
            if len(train_buffer) >= shard_size * 512:  # Approximate tokens per shard
                filename = f"{output_dir}/train/train_shard_{train_shard_idx:04d}.npy"
                np.save(filename, np.array(train_buffer, dtype=np.uint16))
                train_buffer = []
                train_shard_idx += 1
    
    # Save remaining data
    if train_buffer:
        filename = f"{output_dir}/train/train_shard_{train_shard_idx:04d}.npy"
        np.save(filename, np.array(train_buffer, dtype=np.uint16))
        train_shard_idx += 1
    
    if val_buffer:
        filename = f"{output_dir}/val/val_shard_{val_shard_idx:04d}.npy"
        np.save(filename, np.array(val_buffer, dtype=np.uint16))
        val_shard_idx += 1
    
    # Save dataset info
    info = {
        "dataset": dataset_name,
        "total_examples": total_examples,
        "train_examples": total_examples - val_size,
        "val_examples": val_size,
        "train_shards": train_shard_idx,
        "val_shards": val_shard_idx,
        "tokenizer": "gpt2",
    }
    
    with open(f"{output_dir}/dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print()
    print("="*80)
    print("✓ Dataset preparation complete!")
    print("="*80)
    print(f"Train shards: {train_shard_idx}")
    print(f"Val shards: {val_shard_idx}")
    print(f"Output directory: {output_dir}/")
    print()
    print("Sample formatted example:")
    print("-"*80)
    sample = full_data[0]
    print(format_instruction(sample))
    print("-"*80)


if __name__ == "__main__":
    main()

