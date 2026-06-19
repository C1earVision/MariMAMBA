import torch
import yaml
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm
from models.mamba import Mamba
from evaluation.astar_agent import AStarAgent
from data.parser import LevelParser
from config.model_config import MambaConfig
# pyrefly: ignore [missing-import]
from mario_gpt import MarioLM

def load_mamba_model(device: str) -> tuple:
    mamba_config = MambaConfig()

    with open('config/generation_config.yaml', 'r') as f:
        gen_config = yaml.safe_load(f)

    model = Mamba(
        num_tile_types=mamba_config.num_tile_types,
        column_height=mamba_config.column_height,
        tile_embed_dim=mamba_config.tile_embed_dim,
        d_model=mamba_config.d_model,
        n_layers=mamba_config.n_layers,
        d_state=mamba_config.d_state,
        d_conv=mamba_config.d_conv,
        expand=mamba_config.expand,
        dropout=0.0,
        max_seq_len=mamba_config.max_seq_len,
        columns_per_token=mamba_config.columns_per_token,
    ).to(device)

    mamba_checkpoint = torch.load(gen_config['models']['mamba_path'], map_location=device)
    model.load_state_dict(mamba_checkpoint['model_state_dict'])
    model.eval()


    sampling_params = {
        'temperature': gen_config['generation'].get('temperature', 0.8),
        'cfg_scale': gen_config['generation'].get('cfg_scale', 2.0),
        'top_k': gen_config['generation'].get('top_k', 20),
        'top_p': gen_config['generation'].get('top_p', 1.0)
    }

    return model, sampling_params


def count_attributes_in_patch(patch: np.ndarray, model: Mamba) -> List[float]:
    
    total_counts = np.zeros(3)
    device = next(model.parameters()).device
    for col_idx in range(patch.shape[1]):
        column = patch[:, col_idx]
        prev_column = patch[:, col_idx - 1] if col_idx > 0 else None
        
        col_tensor = torch.from_numpy(column).to(device)
        prev_tensor = torch.from_numpy(prev_column).to(device) if prev_column is not None else None
        
        col_attrs = model._count_column_attributes(col_tensor, prev_tensor)
        total_counts += col_attrs.cpu().numpy()
    return total_counts.tolist()


def generate_mamba_samples(
    model,
    targets: List[List[int]],
    samples_per_target: int,
    sampling_params: Dict,
    cfg_scale: float,
    device: str,
    desc: str
) -> Tuple[Dict, float]:
    results = {}
    patch_width = 32
    times = []
    
    for target in targets:
        target_str = str(target)
        results[target_str] = []
        
        attr_tensor = torch.tensor(target).float().to(device)
        
        for _ in tqdm(range(samples_per_target), desc=f"{desc} {target_str}"):
            start_time = time.time()
            with torch.no_grad():
                generated_columns = model.generate(
                    num_columns=patch_width,
                    attributes=attr_tensor,
                    temperature=sampling_params['temperature'],
                    top_k=sampling_params['top_k'],
                    top_p=sampling_params['top_p'],
                    cfg_scale=cfg_scale,
                    device=device,
                )
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start_time)
                patch = generated_columns.cpu().numpy().T
                results[target_str].append(patch)
                
    avg_time = sum(times) / len(times) if times else 0.0
    return results, avg_time


def target_to_mariogpt_prompt(target: List[int]) -> str:
    enemies, gaps, pipes = target
    prompts = []
    
    if enemies == 0: prompts.append("no enemies")
    elif enemies < 3: prompts.append("little enemies")
    else: prompts.append("many enemies")
    
    if gaps == 0: prompts.append("no gaps")
    elif gaps < 3: prompts.append("little gaps")
    else: prompts.append("many gaps")
    
    if pipes == 0: prompts.append("no pipes")
    else: prompts.append("many pipes")
    
    return ", ".join(prompts)


def load_mariogpt_model(device: str):
    print("\nLoading MarioGPT Baseline...")
    mario_lm = MarioLM().to(device)
    return mario_lm


def generate_mariogpt_samples(
    mario_lm,
    targets: List[List[int]],
    samples_per_target: int,
    device: str
) -> Tuple[Dict, float]:
    results = {}
    patch_width = 32
    times = []
    
    for target in targets:
        target_str = str(target)
        results[target_str] = []
        prompt = target_to_mariogpt_prompt(target)
        
        print(f"MarioGPT sampling for: {prompt}")

        parser = LevelParser()
        
        for _ in tqdm(range(samples_per_target), desc=f"MarioGPT {target_str}"):
            start_time = time.time()
            out = mario_lm.sample(
                prompts=[prompt],
                num_steps=patch_width,
                temperature=0.8,
                use_tqdm=False
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start_time)

            level_str = out.level[0]

            level_array = parser.parse_level_list(level_str.split('\n'))
            results[target_str].append(level_array)
            
    avg_time = sum(times) / len(times) if times else 0.0
    return results, avg_time


def evaluate_controllability(samples: Dict, model: Mamba) -> Dict:
    results = {}
    all_errors = []
    all_exact_matches = []

    for target_str, patches in samples.items():
        if target_str == 'overall': continue
        target = eval(target_str)
        actual_counts = []
        exact_matches = 0
        for patch in patches:
            actual = count_attributes_in_patch(patch, model)
            actual_counts.append(actual)

            if np.allclose(actual, target, atol=0.1):
                exact_matches += 1
        
        actual_counts = np.array(actual_counts)
        mean_actual = np.mean(actual_counts, axis=0)
        mae = np.mean(np.abs(actual_counts - target), axis=0)
        accuracy = (exact_matches / len(patches)) if patches else 0.0
        
        results[target_str] = {
            'target': target,
            'mean_actual': mean_actual.tolist(),
            'mae': mae.tolist(),
            'accuracy': accuracy
        }
        all_errors.append(mae)
        all_exact_matches.append(accuracy)

    results['overall'] = {
        'mean_mae': np.mean(all_errors, axis=0).tolist(),
        'total_mae': np.mean(all_errors),
        'accuracy': np.mean(all_exact_matches)
    }
    return results

def evaluate_playability(samples: Dict) -> Dict:
    results = {}
    for target_str, patches in samples.items():
        if target_str == 'overall': continue
        playable_count = 0
        for patch in patches:
            agent = AStarAgent(patch)
            res = agent.find_path()
            if res['playable']:
                playable_count += 1
        results[target_str] = {'playability_rate': playable_count / len(patches) if patches else 0}

    rates = [r['playability_rate'] for k, r in results.items()]
    results['overall'] = {'playability_rate': np.mean(rates)}
    return results


def print_comparison_table(name: str, results: Dict):
    print(f"\n--- {name} ---")
    print(f"{'Target':<12} | {'Actual (Mean)':<20} | {'MAE':<15} | {'Accuracy':<10} | {'Playable'}")
    for target_str, data in results['controllability'].items():
        if target_str == 'overall': continue
        t = data['target']
        a = data['mean_actual']
        m = data['mae']
        acc = data['accuracy']
        p = results['playability'][target_str]['playability_rate']
        print(f"[{t[0]},{t[1]},{t[2]}]".ljust(12) + f" | [{a[0]:.1f},{a[1]:.1f},{a[2]:.1f}]".ljust(20) + f" | [{m[0]:.1f},{m[1]:.1f},{m[2]:.1f}]".ljust(15) + f" | {acc:.1%}    | {p:.1%}")
    print(f"OVERALL MAE: {results['controllability']['overall']['total_mae']:.4f} | OVERALL ACCURACY: {results['controllability']['overall']['accuracy']:.1%} | Playability: {results['playability']['overall']['playability_rate']:.1%}")
    if 'avg_time' in results:
        print(f"Average Inference Time per Sample: {results['avg_time']:.4f} seconds")


def benchmark_mamba_time_vs_length(
    model: Mamba,
    sampling_params: Dict,
    cfg_scale: float,
    level_lengths: List[int],
    num_trials: int,
    device: str,
    desc: str
) -> List[float]:
    """Measure average Mamba inference time to generate levels of varying lengths."""
    avg_times = []
    dummy_target = torch.tensor([1.0, 1.0, 1.0]).to(device)

    for length in level_lengths:
        times = []
        for _ in tqdm(range(num_trials), desc=f"{desc} (length={length})"):
            start_time = time.time()
            with torch.no_grad():
                model.generate(
                    num_columns=length,
                    attributes=dummy_target,
                    temperature=sampling_params['temperature'],
                    top_k=sampling_params['top_k'],
                    top_p=sampling_params['top_p'],
                    cfg_scale=cfg_scale,
                    device=device,
                )
                if device == 'cuda':
                    torch.cuda.synchronize()
            times.append(time.time() - start_time)
        avg_times.append(sum(times) / len(times) if times else 0.0)

    return avg_times


def benchmark_mariogpt_time_vs_length(
    mario_lm,
    level_lengths: List[int],
    num_trials: int,
    device: str
) -> List[float]:
    """Measure average MarioGPT inference time for varying level lengths (num_steps)."""
    avg_times = []
    prompt = "no enemies, no gaps, no pipes"

    for length in level_lengths:
        times = []
        for _ in tqdm(range(num_trials), desc=f"MarioGPT Timing (length={length})"):
            start_time = time.time()
            mario_lm.sample(
                prompts=[prompt],
                num_steps=length,
                temperature=0.8,
                use_tqdm=False
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start_time)
        avg_times.append(sum(times) / len(times) if times else 0.0)

    return avg_times


def print_time_vs_length_table(level_lengths: List[int], timing_results: Dict[str, List[float]]):
    print(f"\n--- Inference Generation Time (s) vs. Level Length ---")
    header = f"{'Level Length':<14}" + "".join(f"{name:<20}" for name in timing_results.keys())
    print(header)
    for i, length in enumerate(level_lengths):
        row = f"{length:<14}" + "".join(f"{timing_results[name][i]:<20.4f}" for name in timing_results.keys())
        print(row)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('config/eval_config.yaml', 'r') as f:
        eval_config = yaml.safe_load(f)['evaluation']
        
    test_targets = eval_config.get('target_attributes', [[0,0,0], [3,0,0], [0,3,0], [3,3,3]])
    samples_per_target = eval_config.get('num_samples_per_target', 5)
    level_lengths = eval_config.get('level_lengths_to_test', [8, 16, 32, 64, 128, 256, 512, 768, 1024])
    timing_trials = eval_config.get('num_timing_trials', 3)

    print(f"Loaded {len(test_targets)} targets from eval_config, generating {samples_per_target} samples per target.")

    model, sampling_params = load_mamba_model(device)
    mario_lm = load_mariogpt_model(device)


    mamba_guided_samples, mamba_guided_time = generate_mamba_samples(model, test_targets, samples_per_target, sampling_params, sampling_params['cfg_scale'], device, "Mamba (Guided)")
    mamba_guided_res = {
        'controllability': evaluate_controllability(mamba_guided_samples, model),
        'playability': evaluate_playability(mamba_guided_samples),
        'avg_time': mamba_guided_time
    }


    mamba_null_samples, mamba_null_time = generate_mamba_samples(model, test_targets, samples_per_target, sampling_params, 1.0, device, "Mamba (Null)")
    mamba_null_res = {
        'controllability': evaluate_controllability(mamba_null_samples, model),
        'playability': evaluate_playability(mamba_null_samples),
        'avg_time': mamba_null_time
    }


    mariogpt_samples, mariogpt_time = generate_mariogpt_samples(mario_lm, test_targets, samples_per_target, device)
    mariogpt_res = {
        'controllability': evaluate_controllability(mariogpt_samples, model),
        'playability': evaluate_playability(mariogpt_samples),
        'avg_time': mariogpt_time
    }

    print("\n" + "="*80)
    print(f"{'FINAL COMPARISON REPORT':^80}")
    print("="*80)
    print_comparison_table("Mamba (Conditional + CFG)", mamba_guided_res)
    print_comparison_table("Mamba (No Guidance/Baseline)", mamba_null_res)
    print_comparison_table("MarioGPT (Text Prompts)", mariogpt_res)

    print("\n" + "="*80)
    print(f"{'INFERENCE TIME vs. LEVEL LENGTH':^80}")
    print("="*80)
    print(f"Benchmarking level lengths {level_lengths} with {timing_trials} trials each.")
    timing_results = {
        'Mamba (Guided)': benchmark_mamba_time_vs_length(model, sampling_params, sampling_params['cfg_scale'], level_lengths, timing_trials, device, "Mamba (Guided)"),
        'Mamba (Null)': benchmark_mamba_time_vs_length(model, sampling_params, 1.0, level_lengths, timing_trials, device, "Mamba (Null)"),
        'MarioGPT': benchmark_mariogpt_time_vs_length(mario_lm, level_lengths, timing_trials, device)
    }
    print_time_vs_length_table(level_lengths, timing_results)
    

    names = ["Mamba (Guided)", "Mamba (Null)", "MarioGPT"]
    maes = [mamba_guided_res['controllability']['overall']['total_mae'], 
            mamba_null_res['controllability']['overall']['total_mae'],
            mariogpt_res['controllability']['overall']['total_mae']]
    plays = [mamba_guided_res['playability']['overall']['playability_rate'],
             mamba_null_res['playability']['overall']['playability_rate'],
             mariogpt_res['playability']['overall']['playability_rate']]
    times = [mamba_guided_res['avg_time'],
             mamba_null_res['avg_time'],
             mariogpt_res['avg_time']]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.bar(names, maes, color=['blue', 'gray', 'green'], alpha=0.7)
    ax1.set_title('Overall Controllability (Lower is Better)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error (MAE)')
    ax1.grid(True, axis='y', alpha=0.3)
    
    ax2.bar(names, plays, color=['blue', 'gray', 'green'], alpha=0.7)
    ax2.set_title('Overall Playability (Higher is Better)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Playability Rate')
    ax2.grid(True, axis='y', alpha=0.3)

    ax3.bar(names, times, color=['blue', 'gray', 'green'], alpha=0.7)
    ax3.set_title('Inference Generation Time (Lower is Better)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Avg Time per Sample (seconds)')
    ax3.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/visualizations/baseline_comparison.png')
    print(f"\nComparison plot saved to output/visualizations/baseline_comparison.png")

    fig2, ax4 = plt.subplots(figsize=(8, 5))
    colors = {'Mamba (Guided)': 'blue', 'Mamba (Null)': 'gray', 'MarioGPT': 'green'}
    for series_name, series_times in timing_results.items():
        ax4.plot(level_lengths, series_times, marker='o', label=series_name, color=colors.get(series_name))
    ax4.set_title('Inference Generation Time vs. Level Length', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Level Length (columns)')
    ax4.set_ylabel('Avg Time per Sample (seconds)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/visualizations/time_vs_length_comparison.png')
    print(f"\nTime vs. Length plot saved to output/visualizations/time_vs_length_comparison.png")