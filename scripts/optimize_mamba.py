import torch
import numpy as np
import glob
import random
# pyrefly: ignore [missing-import]
import optuna
from torch.utils.data import DataLoader

from models.mamba import Mamba
from training.mamba_trainer import MambaTrainer
from data.dataset import ColumnSequenceDataset
from data.parser import LevelParser
from config.model_config import MambaConfig
from config.training_config import MambaTrainingConfig

import sys
import io
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = LevelParser()
model_config = MambaConfig()
base_train_config = MambaTrainingConfig()


raw_data = []
for filepath in sorted(glob.glob("./dataset/*.txt")):
    with open(filepath, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
        raw_data.append(content)

levels = [parser.parse_level_list(lines) for lines in raw_data]

full_dataset = ColumnSequenceDataset(
    levels=levels,
    max_seq_len=base_train_config.max_seq_len,
    stride=base_train_config.stride,
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

def objective(trial):

    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.4)
    ema_decay = trial.suggest_float("ema_decay", 0.9, 0.9999)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


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
        num_attributes=3,
        columns_per_token=model_config.columns_per_token,
    ).to(device)


    trainer = MambaTrainer(
        model=model,
        learning_rate=lr,
        weight_decay=weight_decay,
        device=device
    )
    

    trainer.ema.decay = ema_decay



    best_val_loss = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100, 
        save_interval=1000,
        save_path=f'checkpoints/hpo_trial_{trial.number}.pth',
    )

    return best_val_loss

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MAMBA HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("=" * 70)
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    try:
        study.optimize(objective, n_trials=30)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"  Value: {study.best_value:.4f}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    

    os.makedirs('output/hpo', exist_ok=True)
    import pandas as pd
    study.trials_dataframe().to_csv('output/hpo/optuna_results.csv', index=False)
    print("\nFull results saved to output/hpo/optuna_results.csv")
