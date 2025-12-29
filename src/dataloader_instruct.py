"""
Simple dataloader for instruction fine-tuning dataset.
Adapted from dataloader.py for instruction-formatted data.
"""

import os
import numpy as np
import torch


class InstructDataLoader:
    """Dataloader for instruction fine-tuning shards"""
    
    def __init__(self, B, T, process_rank, num_processes, split='train', data_dir='data_instruct'):
        """
        Args:
            B: Batch size
            T: Sequence length (context length)
            process_rank: Rank of this process in DDP
            num_processes: Total number of processes
            split: 'train' or 'val'
            data_dir: Directory containing instruction data shards
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        
        assert split in {'train', 'val'}
        
        # Get all shard filepaths
        data_path = os.path.join(data_dir, split)
        shard_filenames = sorted([
            f for f in os.listdir(data_path) 
            if f.endswith('.npy')
        ])
        self.shard_filepaths = [
            os.path.join(data_path, f) for f in shard_filenames
        ]
        
        assert len(self.shard_filepaths) > 0, f"No shards found in {data_path}"
        
        if self.process_rank == 0:
            print(f"Found {len(self.shard_filepaths)} {split} shards")
        
        # Initialize state
        self.current_shard_idx = 0
        self.tokens = None
        self.current_position = 0
        
        # Load first shard
        self.reset()
    
    def reset(self):
        """Reset to beginning of dataset"""
        self.current_shard_idx = 0
        self.current_position = self.B * self.T * self.process_rank
        self.load_shard(self.current_shard_idx)
    
    def load_shard(self, shard_idx):
        """Load a specific shard"""
        self.current_shard_idx = shard_idx
        shard_path = self.shard_filepaths[self.current_shard_idx]
        self.tokens = np.load(shard_path).astype(np.int32)
        
        # Set position for this process
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        """Get next batch of data"""
        B, T = self.B, self.T
        
        # Get batch for this process
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        
        # If we don't have enough tokens, move to next shard
        if len(buf) < B*T + 1:
            # Move to next shard
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_filepaths)
            self.load_shard(self.current_shard_idx)
            buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        
        # Create input and target tensors
        x = torch.from_numpy(buf[:-1].reshape(B, T)).long()
        y = torch.from_numpy(buf[1:].reshape(B, T)).long()
        
        # Advance position for all processes
        self.current_position += B * T * self.num_processes
        
        # If we're near the end of the shard, move to next one
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_filepaths)
            self.load_shard(self.current_shard_idx)
        
        return x, y
    
    def get_state(self):
        """Get current state for checkpointing"""
        return {
            'current_shard_idx': self.current_shard_idx,
            'current_position': self.current_position,
        }
    
    def set_state(self, state):
        """Restore state from checkpoint"""
        self.current_shard_idx = state['current_shard_idx']
        self.load_shard(self.current_shard_idx)
        self.current_position = state['current_position']

