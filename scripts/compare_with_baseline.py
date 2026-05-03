import torch
import yaml
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
    ).to(device)

    mamba_checkpoint = torch.load(gen_config['models']['mamba_path'], map_location=device)
    model.load_state_dict(mamba_checkpoint['model_state_dict'])
    model.eval()

    # Load sampling params
    sampling_params = {
        'temperature': gen_config['generation'].get('temperature', 0.8),
        'cfg_scale': gen_config['generation'].get('cfg_scale', 2.0),
        'top_k': gen_config['generation'].get('top_k', 20),
        'top_p': gen_config['generation'].get('top_p', 1.0)
    }

    return model, sampling_params


def count_attributes_in_patch(patch: np.ndarray, model: Mamba) -> List[float]:
    """
    Count total attributes in a generated patch.
    patch: [H, W] array of tile indices
    """
    total_counts = np.zeros(3)
    device = next(model.parameters()).device
    for col_idx in range(patch.shape[1]):
        column = patch[:, col_idx]
        col_attrs = model._count_column_attributes(torch.from_numpy(column).to(device))
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
) -> Dict:
    results = {}
    patch_width = 32
    
    for target in targets:
        target_str = str(target)
        results[target_str] = []
        
        attr_tensor = torch.tensor(target).float().to(device)
        
        for _ in tqdm(range(samples_per_target), desc=f"{desc} {target_str}"):
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
                patch = generated_columns.cpu().numpy().T
                results[target_str].append(patch)
                
    return results


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


def generate_mariogpt_samples(
    targets: List[List[int]],
    samples_per_target: int,
    device: str
) -> Dict:
    print("\nLoading MarioGPT Baseline...")
    mario_lm = MarioLM().to(device)
    results = {}
    patch_width = 32
    
    for target in targets:
        target_str = str(target)
        results[target_str] = []
        prompt = target_to_mariogpt_prompt(target)
        
        print(f"MarioGPT sampling for: {prompt}")
        # MarioGPT generates strings, we need to parse them to indices
        # We'll use a local parser to match indices
        parser = LevelParser()
        
        # We generate one by one to avoid OOM or complex batching
        for _ in tqdm(range(samples_per_target), desc=f"MarioGPT {target_str}"):
            out = mario_lm.sample(
                prompts=[prompt],
                num_steps=patch_width,
                temperature=0.8,
                use_tqdm=False
            )
            # out is a SampleOutput object
            level_str = out.level[0]
            # Convert string to indices [H, W]
            level_array = parser.parse_level_list(level_str.split('\n'))
            results[target_str].append(level_array)
            
    return results


def evaluate_controllability(samples: Dict, model: Mamba) -> Dict:
    results = {}
    all_errors = []

    for target_str, patches in samples.items():
        if target_str == 'overall': continue
        target = eval(target_str)
        actual_counts = []
        for patch in patches:
            actual = count_attributes_in_patch(patch, model)
            actual_counts.append(actual)
        
        actual_counts = np.array(actual_counts)
        mean_actual = np.mean(actual_counts, axis=0)
        mae = np.mean(np.abs(actual_counts - target), axis=0)
        
        results[target_str] = {
            'target': target,
            'mean_actual': mean_actual.tolist(),
            'mae': mae.tolist()
        }
        all_errors.append(mae)

    results['overall'] = {
        'mean_mae': np.mean(all_errors, axis=0).tolist(),
        'total_mae': np.mean(all_errors)
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
    print(f"{'Target':<12} | {'Actual (Mean)':<20} | {'MAE':<15} | {'Playable'}")
    for target_str, data in results['controllability'].items():
        if target_str == 'overall': continue
        t = data['target']
        a = data['mean_actual']
        m = data['mae']
        p = results['playability'][target_str]['playability_rate']
        print(f"[{t[0]},{t[1]},{t[2]}]".ljust(12) + f" | [{a[0]:.1f},{a[1]:.1f},{a[2]:.1f}]".ljust(20) + f" | [{m[0]:.1f},{m[1]:.1f},{m[2]:.1f}]".ljust(15) + f" | {p:.1%}")
    print(f"OVERALL MAE: {results['controllability']['overall']['total_mae']:.4f} | Playability: {results['playability']['overall']['playability_rate']:.1%}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_targets = [[0,0,0], [3,0,0], [0,3,0], [3,3,3]]
    samples_per_target = 5 # Small count for speed

    model, sampling_params = load_mamba_model(device)

    # 1. Our Model (Guided)
    mamba_guided_samples = generate_mamba_samples(model, test_targets, samples_per_target, sampling_params, sampling_params['cfg_scale'], device, "Mamba (Guided)")
    mamba_guided_res = {
        'controllability': evaluate_controllability(mamba_guided_samples, model),
        'playability': evaluate_playability(mamba_guided_samples)
    }

    # 2. Our Model (No Guidance)
    mamba_null_samples = generate_mamba_samples(model, test_targets, samples_per_target, sampling_params, 1.0, device, "Mamba (Null)")
    mamba_null_res = {
        'controllability': evaluate_controllability(mamba_null_samples, model),
        'playability': evaluate_playability(mamba_null_samples)
    }

    # 3. MarioGPT
    mariogpt_samples = generate_mariogpt_samples(test_targets, samples_per_target, device)
    mariogpt_res = {
        'controllability': evaluate_controllability(mariogpt_samples, model),
        'playability': evaluate_playability(mariogpt_samples)
    }

    print("\n" + "="*80)
    print(f"{'FINAL COMPARISON REPORT':^80}")
    print("="*80)
    print_comparison_table("Mamba (Conditional + CFG)", mamba_guided_res)
    print_comparison_table("Mamba (No Guidance/Baseline)", mamba_null_res)
    print_comparison_table("MarioGPT (Text Prompts)", mariogpt_res)
    
    # Save a comparison plot
    names = ["Mamba (Guided)", "Mamba (Null)", "MarioGPT"]
    maes = [mamba_guided_res['controllability']['overall']['total_mae'], 
            mamba_null_res['controllability']['overall']['total_mae'],
            mariogpt_res['controllability']['overall']['total_mae']]
    plays = [mamba_guided_res['playability']['overall']['playability_rate'],
             mamba_null_res['playability']['overall']['playability_rate'],
             mariogpt_res['playability']['overall']['playability_rate']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(names, maes, color=['blue', 'gray', 'green'], alpha=0.7)
    ax1.set_title('Overall Controllability (Lower is Better)')
    ax1.set_ylabel('Mean Absolute Error (MAE)')
    
    ax2.bar(names, plays, color=['blue', 'gray', 'green'], alpha=0.7)
    ax2.set_title('Overall Playability (Higher is Better)')
    ax2.set_ylabel('Playability Rate')
    
    plt.tight_layout()
    plt.savefig('output/visualizations/baseline_comparison.png')
    print(f"\nComparison plot saved to output/visualizations/baseline_comparison.png")
