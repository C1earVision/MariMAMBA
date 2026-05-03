import torch
import yaml
from models.autoencoder import Autoencoder
from models.mamba import ConditionalMamba
from generation.stitcher import PatchStitcher
from models.latent_normalizer import LatentNormalizer
from evaluation.difficulty_evaluator import PatchDifficultyEvaluator
from data.parser import LevelParser
from config.model_config import (
    AutoencoderConfig,
    MambaConfig,
    NormalizerConfig
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

ae_config = AutoencoderConfig()
mamba_config = MambaConfig()
normalizer_config = NormalizerConfig()

with open('config/generation_config.yaml', 'r') as f:
    gen_config = yaml.safe_load(f)

autoencoder = Autoencoder(
    num_tile_types=ae_config.num_tile_types,
    embedding_dim=ae_config.embedding_dim,
    latent_dim=ae_config.latent_dim,
    patch_height=ae_config.patch_height,
    patch_width=ae_config.patch_width
)
ae_checkpoint = torch.load(gen_config['models']['autoencoder_path'], map_location=device)
autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
autoencoder.to(device)
autoencoder.eval()
print("Autoencoder loaded")

model = ConditionalMamba(
    latent_dim=mamba_config.latent_dim,
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
print("Mamba model loaded")

normalizer = LatentNormalizer(target_norm=normalizer_config.norm)
normalizer.load(gen_config['models']['normalizer_path'])

parser = LevelParser()
difficulty_evaluator = PatchDifficultyEvaluator(parser)
stitcher = PatchStitcher()

print("All components initialized")
print("\n" + "="*70)
print("Starting Difficulty Evaluation Comparison")
print("="*70)

target_difficulties = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
num_samples = 10
temperature = gen_config['generation']['temperature']

all_results = {}
for target_diff in target_difficulties:
    print(f"\nGenerating {num_samples} patches with target difficulty = {target_diff}...")
    actual_scores = []

    for sample_idx in range(num_samples):
        difficulties = torch.tensor([target_diff], dtype=torch.float32)
        latent = model.generate(
            num_patches=1,
            difficulties=difficulties,
            initial_latent=None,
            temperature=temperature,
            device=device,
        )
        latent = latent.squeeze(0)
        latent_denorm = normalizer.denormalize(latent)

        with torch.no_grad():
            latent_denorm = latent_denorm.to(device)
            decoded_logits = autoencoder.decoder(latent_denorm)
            if decoded_logits.dim() == 4:
                patch = torch.argmax(decoded_logits, dim=1)
            else:
                patch = torch.argmax(decoded_logits, dim=-1)
            patch = patch.cpu().numpy()[0]

        eval_result = difficulty_evaluator.evaluate_patch(
            patch,
            metadata={'target_difficulty': target_diff, 'sample_idx': sample_idx}
        )
        actual_scores.append(eval_result['scores']['difficulty_score'])

    import numpy as np
    mean_score = np.mean(actual_scores)
    std_score = np.std(actual_scores)
    error = abs(mean_score - target_diff)
    all_results[target_diff] = {
        'mean': mean_score, 'std': std_score, 'error': error, 'scores': actual_scores
    }
    print(f"  Target: {target_diff:.2f} | Actual: {mean_score:.3f} ± {std_score:.3f} | Error: {error:.3f}")

print("\nEvaluation complete!")
print(f"Results saved to: output/visualizations/difficulty_evaluation.png")
