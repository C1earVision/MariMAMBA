import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import torch
import math
import argparse
import yaml
import os
import numpy as np

from models.mamba import Mamba
from generation.renderer import save_level_image
from data.parser import LevelParser
from config.model_config import MambaConfig
from config.training_config import MambaTrainingConfig
from map_fixer_system.map_fixer import create_difficulty_schedule

import sys
import io
warnings.filterwarnings('ignore', category=UserWarning)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ─── Arguments ───────────────────────────────────────────────────────
arg_parser = argparse.ArgumentParser(description='Generate levels with Mamba')
arg_parser.add_argument('--attributes', type=str, default=None, help='Comma-separated target counts [enemies, gaps, pipes]')
arg_parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature')
arg_parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling (0 = disabled)')
arg_parser.add_argument('--top_p', type=float, default=None, help='Top-p nucleus sampling (1.0 = disabled)')
arg_parser.add_argument('--columns', type=int, default=None, help='Number of columns per level')
arg_parser.add_argument('--num_levels', type=int, default=None, help='Number of levels to generate')
arg_parser.add_argument('--cfg_scale', type=float, default=None, help='CFG scale (1.0 = no guidance)')
args = arg_parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_config = MambaConfig()
parser = LevelParser()

print("=" * 70)
print("GENERATING LEVELS (Mamba)")
print("=" * 70)
print(f"Device: {device}")

# ─── Load Config ─────────────────────────────────────────────────────
with open('config/generation_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

m_cfg = MambaConfig()
train_cfg = MambaTrainingConfig()

columns_per_level = args.columns if args.columns is not None else config['generation']['columns_per_level']
num_levels = args.num_levels if args.num_levels is not None else config['generation']['num_levels']

if args.attributes:
    attribute_target = [float(x) for x in args.attributes.split(',')]
else:
    attribute_target = config['generation'].get('attribute_target', [2.0, 0.0, 1.0])
temperature = args.temperature if args.temperature is not None else config['generation']['temperature']
top_k = args.top_k if args.top_k is not None else config['generation'].get('top_k', 0)
top_p = args.top_p if args.top_p is not None else config['generation'].get('top_p', 1.0)
cfg_scale = args.cfg_scale if args.cfg_scale is not None else config['generation'].get('cfg_scale', 3.0)

model_path = config['models'].get('mamba_path')
if not model_path:
    model_path = train_cfg.save_path.replace('.pth', '_best.pth')
output_dir = config['output']['directory']

# ─── Load Model ──────────────────────────────────────────────────────
model = Mamba(
    num_tile_types=m_cfg.num_tile_types,
    column_height=m_cfg.column_height,
    tile_embed_dim=m_cfg.tile_embed_dim,
    d_model=m_cfg.d_model,
    n_layers=m_cfg.n_layers,
    d_state=m_cfg.d_state,
    d_conv=m_cfg.d_conv,
    expand=m_cfg.expand,
    dropout=0.0,
    max_seq_len=m_cfg.max_seq_len,
    num_attributes=m_cfg.num_attributes,
    columns_per_token=m_cfg.columns_per_token
).to(device)

checkpoint = torch.load(model_path, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
print("Mamba model loaded")

# ─── Generate Levels ─────────────────────────────────────────────────
generated_levels = []

for level_idx in range(num_levels):
    print(f"\n{'='*70}")
    print(f"Generating Level {level_idx + 1}/{num_levels}")
    print(f"{'='*70}")
    print(f"  Columns: {columns_per_level}")
    print(f"  Attributes: {attribute_target}")
    print(f"  Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
    print(f"  CFG scale: {cfg_scale}")

    # For dynamic counts, we pass the initial target as a single vector
    # The model's generate() method will handle the decrementing
    attr_tensor = torch.tensor(attribute_target).to(device)
    
    # Generate columns
    generated_columns = model.generate(
        num_columns=columns_per_level,
        attributes=attr_tensor,
        initial_column=None,  # Default ground floor
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_scale=cfg_scale,
        device=device,
    )

    # generated_columns: [num_columns, H] -> transpose to [H, num_columns]
    level_array = generated_columns.cpu().numpy().T  # [H, W]

    print(f"Generated level shape: {level_array.shape}")

    # Decode to text
    level_str = parser.decode_level(level_array)
    print(f"\n--- Generated Level {level_idx + 1} ---")
    print(level_str)

    generated_levels.append((level_str, level_array))

# ─── Save Results ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"GENERATION COMPLETE! Generated {len(generated_levels)} levels")
print(f"{'='*70}")

os.makedirs(output_dir, exist_ok=True)

for i, (level_str, level_array) in enumerate(generated_levels):
    # Save text
    txt_path = os.path.join(output_dir, f'column_level_{i+1}.txt')
    with open(txt_path, 'w') as f:
        f.write(level_str)
    print(f'Saved level {i+1} text to {txt_path}')

    # Save PNG
    png_path = os.path.join(output_dir, f'column_level_{i+1}.png')
    save_level_image(level_array, png_path, tile_size=16)
    print(f'Saved level {i+1} image to {png_path}')
