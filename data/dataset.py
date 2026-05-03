import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Sequence, List, Dict
from data.parser import LevelParser
from evaluation.difficulty_evaluator import PatchDifficultyEvaluator


class ColumnSequenceDataset(Dataset):
    def __init__(
        self,
        levels: List[np.ndarray],
        max_seq_len: int = 128,
        stride: int = 1,
        parser: Optional[LevelParser] = None,
        num_attributes: int = 3,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_attributes = num_attributes
        self.parser = parser
        
        self.sequences = []
        
        # Define tile mappings for attribute counting
        # [enemies, gaps, pipes]
        self.enemy_tiles = {parser.tile_to_idx[t] for t in ['E', 'B'] if t in parser.tile_to_idx}
        self.pipe_tiles = {parser.tile_to_idx[t] for t in ['['] if t in parser.tile_to_idx}
        self.empty_tile = parser.tile_to_idx['-']

        for level_idx, level in enumerate(levels):
            H, W = level.shape
            columns = level.T  # [W, H] 
            
            # 1. Pre-compute attributes for each column in this level
            level_attrs = self._get_level_attributes(columns) # [W, num_attributes]

            if W < 2:
                continue
            window_size = min(max_seq_len + 1, W)
            for start in range(0, W - 1, stride):
                end = min(start + window_size, W)
                if end - start < 2:
                    continue

                chunk_cols = torch.from_numpy(columns[start:end]).long()
                chunk_attrs = level_attrs[start:end] # [win, 4]

                # 2. Compute "Remaining Counts" for this chunk
                # For each step t, the conditioning is the sum of attributes from t to the end of the chunk
                # Shape: [win, 4]
                remaining_counts = np.flip(np.cumsum(np.flip(chunk_attrs, axis=0), axis=0), axis=0).copy()
                
                input_seq = chunk_cols[:-1]
                target_seq = chunk_cols[1:]
                # Conditioning for predicting target_seq[t] is remaining_counts[t]
                cond_seq = torch.from_numpy(remaining_counts[:-1]).float()

                seq_len = input_seq.shape[0]
                pad_len = max_seq_len - seq_len

                if pad_len > 0:
                    H_dim = input_seq.shape[1]
                    input_seq = torch.cat([
                        input_seq,
                        torch.zeros(pad_len, H_dim, dtype=torch.long)
                    ], dim=0)
                    target_seq = torch.cat([
                        target_seq,
                        torch.zeros(pad_len, H_dim, dtype=torch.long)
                    ], dim=0)
                    cond_seq = torch.cat([
                        cond_seq,
                        torch.zeros(pad_len, num_attributes, dtype=torch.float32)
                    ], dim=0)

                self.sequences.append((input_seq, cond_seq, target_seq, seq_len))

        print(f"ColumnSequenceDataset: {len(self.sequences)} sequences from "
              f"{len(levels)} levels (stride={stride}, max_seq_len={max_seq_len})")

    def _get_level_attributes(self, columns: np.ndarray) -> np.ndarray:
        W, H = columns.shape
        attrs = np.zeros((W, 3), dtype=np.float32)
        
        for i in range(W):
            col = columns[i]
            # Enemies
            if any(tile in self.enemy_tiles for tile in col):
                attrs[i, 0] = 1.0
            
            # Gaps (check bottom tile)
            if col[H-1] == self.empty_tile:
                attrs[i, 1] = 1.0
            
            # Pipes
            if any(tile in self.pipe_tiles for tile in col):
                attrs[i, 2] = 1.0
                
        return attrs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, cond_seq, target_seq, seq_len = self.sequences[idx]
        return input_seq, cond_seq, target_seq, seq_len
