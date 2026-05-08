import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm

from models.mamba import Mamba
from data.parser import LevelParser
from config.model_config import MambaConfig

import sys
import io
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def count_attributes(patch: np.ndarray, model: Mamba) -> np.ndarray:
    
    total_counts = np.zeros(3)
    device = next(model.parameters()).device
    for col_idx in range(patch.shape[1]):
        column = patch[:, col_idx]
        col_attrs = model._count_column_attributes(torch.from_numpy(column).to(device))
        total_counts += col_attrs.cpu().numpy()
    return total_counts

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model_config = MambaConfig()

    with open('config/generation_config.yaml', 'r') as f:
        gen_config = yaml.safe_load(f)

    with open('config/eval_config.yaml', 'r') as f:
        eval_config = yaml.safe_load(f)['evaluation']

    # ─── Load Model ──────────────────────────────────────────────────────
    raw_ckpt = torch.load(gen_config['models']['mamba_path'], map_location=device)
    state_dict = raw_ckpt.get('model_state_dict', raw_ckpt) if isinstance(raw_ckpt, dict) and 'model_state_dict' in raw_ckpt else raw_ckpt

    # Auto-detect architecture from checkpoint
    ckpt_d_state = state_dict['layers.0.ssm.A_log'].shape[-1]
    ckpt_max_seq_len = state_dict['pos_embedding'].shape[1]

    if ckpt_d_state != model_config.d_state or ckpt_max_seq_len != model_config.max_seq_len:
        print(f"  ⚠ Using checkpoint values: d_state={ckpt_d_state}, max_seq_len={ckpt_max_seq_len}")

    model = Mamba(
        num_tile_types=model_config.num_tile_types,
        column_height=model_config.column_height,
        tile_embed_dim=model_config.tile_embed_dim,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        d_state=ckpt_d_state,
        d_conv=model_config.d_conv,
        expand=model_config.expand,
        dropout=0.0,
        max_seq_len=ckpt_max_seq_len,
        columns_per_token=model_config.columns_per_token,
        num_attributes=3
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    print("Mamba model loaded")

    # ─── Evaluation Params ───────────────────────────────────────────────
    target_attributes = eval_config.get('target_attributes', [[0,0,0]])
    num_samples = eval_config.get('num_samples_per_target', 5)
    temperature = eval_config.get('temperature', 0.8)
    cfg_scale = eval_config.get('cfg_scale', 3.0)
    top_k = eval_config.get('top_k', 20)
    top_p = eval_config.get('top_p', 1.0)
    columns_per_sample = eval_config.get('columns_per_level', 64)

    print(f"\n{'='*70}")
    print(f"MAMBA ATTRIBUTE EVALUATION")
    print(f"{'='*70}")
    print(f"Targets: {target_attributes}")
    print(f"Samples per target: {num_samples}")
    print(f"Columns per sample: {columns_per_sample}")

    results_summary = {}
    attr_names = ["Enemies", "Gaps", "Pipes"]

    for target in target_attributes:
        target_str = str(target)
        print(f"\nEvaluating target {target_str}...")
        
        target_tensor = torch.tensor(target).float().to(device)
        sample_counts = []

        for _ in tqdm(range(num_samples), desc=f"Target {target_str}"):
            with torch.no_grad():
                generated_columns = model.generate(
                    num_columns=columns_per_sample,
                    attributes=target_tensor,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    cfg_scale=cfg_scale,
                    device=device,
                )
                level_array = generated_columns.cpu().numpy().T
                actual = count_attributes(level_array, model)
                sample_counts.append(actual)

        sample_counts = np.array(sample_counts)
        mean_actual = np.mean(sample_counts, axis=0)
        std_actual = np.std(sample_counts, axis=0)
        mae = np.mean(np.abs(sample_counts - target), axis=0)

        results_summary[target_str] = {
            'target': target,
            'mean': mean_actual,
            'std': std_actual,
            'mae': mae,
            'samples': sample_counts
        }

    # ─── Report ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Target':<15} | {'Actual (Mean)':<25} | {'MAE (Mean)':<20}")
    print(f"{'-'*80}")
    for t_str, data in results_summary.items():
        t = data['target']
        m = data['mean']
        mae = data['mae']
        print(f"{str(t):<15} | [{m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f}]".ljust(43) + f" | [{mae[0]:.1f}, {mae[1]:.1f}, {mae[2]:.1f}]")
    
    overall_mae = np.mean([data['mae'] for data in results_summary.values()], axis=0)
    print(f"{'='*80}")
    print(f"OVERALL MAE: Enemies={overall_mae[0]:.3f}, Gaps={overall_mae[1]:.3f}, Pipes={overall_mae[2]:.3f}")
    print(f"{'='*80}\n")

    # ─── Plotting ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    x = np.arange(len(target_attributes))
    labels = [str(t) for t in target_attributes]
    width = 0.35

    for i, name in enumerate(attr_names):
        targets = [t[i] for t in target_attributes]
        actuals = [results_summary[str(t)]['mean'][i] for t in target_attributes]
        stds = [results_summary[str(t)]['std'][i] for t in target_attributes]

        axes[i].bar(x - width/2, targets, width, label='Target', color='gray', alpha=0.5)
        axes[i].bar(x + width/2, actuals, width, label='Actual', color=['red', 'blue', 'green'][i], alpha=0.7)
        axes[i].errorbar(x + width/2, actuals, yerr=stds, fmt='none', ecolor='black', capsize=5)
        
        axes[i].set_title(f'{name} Control', fontsize=14, fontweight='bold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = eval_config.get('evaluation_path', 'output/visualizations/attribute_evaluation.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Evaluation visualization saved to {save_path}")

if __name__ == '__main__':
    main()
