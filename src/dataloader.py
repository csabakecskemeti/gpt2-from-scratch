import os
import numpy as np
import torch

script_dir = os.path.dirname(__file__)


class DataLoaderLite:
    """ A simple dataloader for FineWebEdu-10B dataset """

    def __init__(self, B, T, process_rank, num_processes, split='train', shuffle=True):
        super().__init__()
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.shuffle = shuffle  # Whether to shuffle shards between epochs
        assert split in {'train', 'val'}
        
        # get the shard filenames
        data_root = os.path.join(script_dir, "../data/edu_fineweb10B")
        shard_filenames = os.listdir(data_root)
        shard_filenames = sorted([filename for filename in shard_filenames if split in filename])
        self.shard_filepaths = [os.path.join(data_root, filename) for filename in shard_filenames]
        assert len(self.shard_filepaths) > 0, f'no shards found for split {split}'
        
        # Store original order for reproducibility
        self.original_shard_filepaths = self.shard_filepaths.copy()
        
        master_process = process_rank == 0
        if master_process:
            print(f'found {len(self.shard_filepaths)} shards for split {split}')
            if shuffle and split == 'train':
                print(f'shuffle enabled: shards will be shuffled between epochs')
        self.reset()

    def load_tokens(self, filepath):
        tokens = torch.tensor(np.load(filepath).astype(np.int32), dtype=torch.long)
        return tokens

    def reset(self):
        # state, init at shard 0
        self.curr_shard = 0
        self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
        self.curr_pos = self.B * self.T * self.process_rank
    
    def shuffle_shards(self):
        """
        Shuffle the order of shards for the next epoch.
        Uses numpy's RNG which is saved/restored in checkpoints for reproducibility.
        Only shuffles training data, not validation.
        """
        if self.shuffle and self.split == 'train':
            # Use numpy's random (which is tracked in checkpoints)
            indices = np.arange(len(self.shard_filepaths))
            np.random.shuffle(indices)
            self.shard_filepaths = [self.shard_filepaths[i] for i in indices]
            master_process = self.process_rank == 0
            if master_process:
                print(f"🔀 Shuffled {len(self.shard_filepaths)} training shards for new epoch")
    
    def reset_epoch(self):
        """
        Reset dataloader for a new epoch with optional shuffling.
        This should be called at the start of each new epoch.
        """
        self.shuffle_shards()
        self.reset()

    def next_batch(self):
        B, T = self.B, self.T
        batch = self.tokens[self.curr_pos : self.curr_pos + B*T + 1]
        x_batch = batch[:-1].view(B, T)
        y_batch = batch[1:].view(B, T)
        self.curr_pos += B * T * self.num_processes
        if self.curr_pos + (B * T + 1) > len(self.tokens):
            self.curr_shard = (self.curr_shard + 1) % len(self.shard_filepaths)
            self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
            self.curr_pos = self.B * self.T * self.process_rank
        return x_batch, y_batch