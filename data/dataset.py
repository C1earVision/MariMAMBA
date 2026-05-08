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
        columns_per_token: int = 1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_attributes = num_attributes
        self.parser = parser
        self.columns_per_token = columns_per_token
        
        self.sequences = []
        


        self.enemy_tiles = {parser.tile_to_idx[t] for t in ['E'] if t in parser.tile_to_idx}
        self.pipe_tiles = {parser.tile_to_idx[t] for t in ['['] if t in parser.tile_to_idx}
        self.empty_tile = parser.tile_to_idx['-']

        for level_idx, level in enumerate(levels):
            H, W = level.shape
            columns = level.T
            level_attrs = self._get_level_attributes(columns)


            K = self.columns_per_token
            num_tokens = W // K
            if num_tokens < 2:
                continue
            


            tokenized_columns = torch.from_numpy(columns[:num_tokens * K]).reshape(num_tokens, K * H).long()

            tokenized_attrs = torch.from_numpy(level_attrs[:num_tokens * K]).reshape(num_tokens, K, 3).sum(dim=1)

            W_tok = num_tokens
            window_size = min(max_seq_len + 1, W_tok)
            
            for start in range(0, W_tok - 1, stride):
                end = min(start + window_size, W_tok)
                if end - start < 2:
                    continue

                chunk_cols = tokenized_columns[start:end]
                chunk_attrs = tokenized_attrs[start:end]



                remaining_counts = torch.flip(torch.cumsum(torch.flip(chunk_attrs, dims=[0]), dim=0), dims=[0])
                
                input_seq = chunk_cols[:-1]
                target_seq = chunk_cols[1:]
                cond_seq = remaining_counts[:-1].float()

                seq_len = input_seq.shape[0]
                pad_len = max_seq_len - seq_len

                if pad_len > 0:
                    H_eff = K * H
                    input_seq = torch.cat([
                        input_seq,
                        torch.zeros(pad_len, H_eff, dtype=torch.long)
                    ], dim=0)
                    target_seq = torch.cat([
                        target_seq,
                        torch.zeros(pad_len, H_eff, dtype=torch.long)
                    ], dim=0)
                    cond_seq = torch.cat([
                        cond_seq,
                        torch.zeros(pad_len, 3, dtype=torch.float32)
                    ], dim=0)

                self.sequences.append((input_seq, cond_seq, target_seq, seq_len))

        print(f"ColumnSequenceDataset: {len(self.sequences)} sequences ({self.columns_per_token} cols/token) from "
              f"{len(levels)} levels")

    def _get_level_attributes(self, columns: np.ndarray) -> np.ndarray:
        W, H = columns.shape
        attrs = np.zeros((W, 3), dtype=np.float32)
        
        for i in range(W):
            col = columns[i]

            if any(tile in self.enemy_tiles for tile in col):
                attrs[i, 0] = 1.0
            

            if col[H-1] == self.empty_tile:
                attrs[i, 1] = 1.0
            

            if any(tile in self.pipe_tiles for tile in col):
                attrs[i, 2] = 1.0
                
        return attrs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, cond_seq, target_seq, seq_len = self.sequences[idx]
        return input_seq, cond_seq, target_seq, seq_len
