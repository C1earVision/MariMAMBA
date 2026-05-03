import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Dict, Tuple

class PatchStitcher:
    def __init__(self,
                 patch_height: int = 14,
                 patch_width: int = 16,
                 stride: int = 2):

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride = stride

    def stitch_patches_to_level(self,
                            patches: np.ndarray,
                            target_width: int = None,
                            ) -> np.ndarray:
        num_patches = len(patches)
        first_patch = patches[0]
        ph, pw = first_patch.shape

        if ph != self.patch_height or pw != self.patch_width:
            self.patch_height = ph
            self.patch_width = pw

        target_width = (num_patches - 1) * self.stride + pw

        level = np.zeros((ph, target_width), dtype=np.int32)

        for i, patch in enumerate(patches):
            if patch.ndim != 2:
                raise ValueError(f"Each patch must be 2D. Got ndim={patch.ndim} at index {i}.")

            ph_i, pw_i = patch.shape
            if ph_i != ph:
                if ph_i > ph:
                    patch = patch[:ph, :]
                else:
                    patch = np.pad(patch, ((0, ph - ph_i), (0, 0)), mode='constant', constant_values=0)

            x_start = i * self.stride

            if i == num_patches - 1:
                x_end = x_start + pw_i
                patch_slice = patch
            else:
                x_end = x_start + self.stride
                patch_slice = patch[:, :self.stride]
            if x_end > target_width:
                overflow = x_end - target_width
                x_end = target_width
                patch_slice = patch_slice[:, :-overflow]
            level[:, x_start:x_end] = patch_slice

        return level