import torch
import numpy as np
import pickle as pkl
import glob
import random
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.mamba import Mamba
from training.mamba_trainer import MambaTrainer
from data.dataset import ColumnSequenceDataset
from data.parser import LevelParser
from evaluation.difficulty_evaluator import PatchDifficultyEvaluator
from config.model_config import MambaConfig
from config.training_config import MambaTrainingConfig

import sys
import io
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

model_config = MambaConfig()
train_config = MambaTrainingConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("MAMBA — DIRECT TILE-LEVEL TRAINING")
print("=" * 70)
print(f"Device: {device}")

parser = LevelParser()
difficulty_evaluator = PatchDifficultyEvaluator(parser)

raw_data = []
for filepath in sorted(glob.glob("./dataset/*.txt")):
    with open(filepath, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
        raw_data.append(content)

print(f"\nLoaded {len(raw_data)} raw levels")

levels = []
for level_lines in raw_data:
    level_array = parser.parse_level_list(level_lines)
    levels.append(level_array)
    
print(f"Parsed {len(levels)} levels")
for i, lvl in enumerate(levels[:3]):
    print(f"  Level {i+1}: shape {lvl.shape} ({lvl.shape[1]} columns)")

print("\n" + "=" * 70)
print("Creating Column Sequence Dataset")
print("=" * 70)

full_dataset = ColumnSequenceDataset(
    levels=levels,
    max_seq_len=train_config.max_seq_len,
    stride=train_config.stride,
    parser=parser,
    num_attributes=3,
    columns_per_token=model_config.columns_per_token,
)

random.seed(42)
all_indices = list(range(len(full_dataset)))
random.shuffle(all_indices)
split = int(0.8 * len(all_indices))
train_indices = all_indices[:split]
val_indices = all_indices[split:]

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

train_loader = DataLoader(
    train_dataset, 
    batch_size=train_config.batch_size, 
    shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)

print(f"  Total sequences: {len(full_dataset)}")
print(f"  Train: {len(train_dataset)} sequences")
print(f"  Val: {len(val_dataset)} sequences")

print("\n" + "=" * 70)
print("Creating Mamba Model")
print("=" * 70)

model = Mamba(
    num_tile_types=model_config.num_tile_types,
    column_height=model_config.column_height,
    tile_embed_dim=model_config.tile_embed_dim,
    d_model=model_config.d_model,
    n_layers=model_config.n_layers,
    d_state=model_config.d_state,
    d_conv=model_config.d_conv,
    expand=model_config.expand,
    dropout=model_config.dropout,
    max_seq_len=model_config.max_seq_len,
    num_attributes=model_config.num_attributes,
    columns_per_token=model_config.columns_per_token,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model created: {num_params:,} parameters")

trainer = MambaTrainer(
    model=model,
    learning_rate=train_config.learning_rate,
    weight_decay=train_config.weight_decay,
    device=device
)

print("\n" + "=" * 70)
print("Training")
print("=" * 70)

trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=train_config.num_epochs,
    save_interval=train_config.save_interval,
    save_path=train_config.save_path
)

import os
os.makedirs('output/visualizations', exist_ok=True)
trainer.plot_losses('output/visualizations/mamba_losses.png')

full_checkpoint = {
    'trainer_state': {
        'epoch_losses': trainer.epoch_losses,
        'val_losses': trainer.val_losses,
    },
    'model_state_dict': model.state_dict(),
    'config': {
        'num_tile_types': model_config.num_tile_types,
        'column_height': model_config.column_height,
        'tile_embed_dim': model_config.tile_embed_dim,
        'd_model': model_config.d_model,
        'n_layers': model_config.n_layers,
        'd_state': model_config.d_state,
        'd_conv': model_config.d_conv,
        'expand': model_config.expand,
        'max_seq_len': model_config.max_seq_len,
        'num_attributes': 3,
        'columns_per_token': model_config.columns_per_token,
        'batch_size': train_config.batch_size,
    }
}

torch.save(full_checkpoint, 'checkpoints/mamba_full_state.pth')
print("Full training state saved")

with open('checkpoints/column_val_loader_info.pkl', 'wb') as f:
    pkl.dump({
        'val_indices': val_indices,
        'batch_size': train_config.batch_size,
    }, f)
print("Val loader info saved")
