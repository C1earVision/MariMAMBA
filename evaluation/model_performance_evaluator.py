import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


class ModelPerformanceEvaluator:    
    def evaluate_difficulty_comparison(self,
                                      mamba_model,
                                      autoencoder,
                                      difficulty_evaluator,
                                      normalizer=None,
                                      target_difficulties: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
                                      num_samples_per_target: int = 5,
                                      temperature: float = 1.0,
                                      device: str = 'cuda',
                                      save_path: Optional[str] = None,
                                      **generate_kwargs) -> Dict:

        print(f"\n{'='*70}")
        print(f"DIFFICULTY EVALUATION COMPARISON")
        print(f"{'='*70}")
        print(f"Target difficulties: {target_difficulties}")
        print(f"Samples per target: {num_samples_per_target}")
        print(f"{'='*70}\n")

        results = {
            'target_difficulties': target_difficulties,
            'evaluations': [],
            'summary': {}
        }

        all_targets = []
        all_actual_scores = []

        for target_diff in target_difficulties:
            print(f"\nGenerating {num_samples_per_target} patches with target difficulty = {target_diff}...")

            target_scores = []
            actual_scores = []
            patches_list = []

            for sample_idx in range(num_samples_per_target):
                difficulties = torch.tensor([target_diff], dtype=torch.float32)

                tokens = mamba_model.generate(
                    num_patches=1,
                    difficulties=difficulties,
                    initial_token=None,
                    temperature=temperature,
                    device=device,
                    **generate_kwargs
                )

                tokens = tokens.squeeze(0)  # [L]

                # Decode tokens back to latent vectors
                if hasattr(autoencoder, 'use_vq') and autoencoder.use_vq:
                    print(f"Generated token: {tokens.item()}")
                    latent_denorm = autoencoder.vq_layer._embedding(tokens.long())
                elif normalizer is not None:
                    latent_denorm = normalizer.denormalize(tokens)
                else:
                    raise ValueError("Non-VQ model requires a normalizer")

                with torch.no_grad():
                    latent_denorm = latent_denorm.to(device)
                    decoded_logits = autoencoder.decoder(latent_denorm)

                    if decoded_logits.dim() == 4:
                        patch = torch.argmax(decoded_logits, dim=1)
                    elif decoded_logits.dim() == 3:
                        patch = decoded_logits.long()
                    elif decoded_logits.dim() == 5:
                        decoded_logits = decoded_logits.squeeze(1)
                        patch = torch.argmax(decoded_logits, dim=1)
                    else:
                        patch = torch.argmax(decoded_logits, dim=-1)

                    patch = patch.cpu().numpy()[0]

                eval_result = difficulty_evaluator.evaluate_patch(
                    patch,
                    metadata={'target_difficulty': target_diff, 'sample_idx': sample_idx}
                )

                difficulty_score = eval_result['scores']['difficulty_score']

                target_scores.append(target_diff)
                actual_scores.append(difficulty_score)
                patches_list.append(patch)

                all_targets.append(target_diff)
                all_actual_scores.append(difficulty_score)

                results['evaluations'].append({
                    'target_difficulty': target_diff,
                    'actual_difficulty': difficulty_score,
                    'sample_idx': sample_idx,
                    'patch': patch,
                    'full_evaluation': eval_result
                })

            mean_actual = np.mean(actual_scores)
            std_actual = np.std(actual_scores)

            results['summary'][target_diff] = {
                'mean_difficulty': mean_actual,
                'std_difficulty': std_actual,
                'target_difficulty': target_diff,
                'error': abs(mean_actual - target_diff),
                'samples': actual_scores
            }

            print(f"  Target: {target_diff:.2f} | "
                  f"Actual Difficulty: {mean_actual:.3f} ± {std_actual:.3f} | "
                  f"Error: {abs(mean_actual - target_diff):.3f}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        target_vals = list(results['summary'].keys())
        mean_vals = [results['summary'][t]['mean_difficulty'] for t in target_vals]
        std_vals = [results['summary'][t]['std_difficulty'] for t in target_vals]

        ax1.errorbar(target_vals, mean_vals, yerr=std_vals,
                    fmt='o', markersize=8, capsize=5, capthick=2,
                    label='Generated (Mean ± Std)', color='blue', alpha=0.7)
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Alignment', alpha=0.5)
        ax1.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Actual Difficulty Score', fontsize=12, fontweight='bold')
        ax1.set_title('Target vs Actual Difficulty (Aggregated)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)

        ax2 = axes[0, 1]
        ax2.scatter(all_targets, all_actual_scores, alpha=0.5, s=50, color='green')
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Alignment', alpha=0.5)
        ax2.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual Difficulty Score', fontsize=12, fontweight='bold')
        ax2.set_title('All Individual Samples', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)

        ax3 = axes[1, 0]
        errors = [results['summary'][t]['error'] for t in target_vals]
        ax3.bar(range(len(target_vals)), errors, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(target_vals)))
        ax3.set_xticklabels([f'{t:.1f}' for t in target_vals])
        ax3.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
        ax3.set_title('Prediction Error by Target', fontsize=14, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)

        ax4 = axes[1, 1]
        data_for_box = [results['summary'][t]['samples'] for t in target_vals]
        bp = ax4.boxplot(data_for_box, positions=range(len(target_vals)),
                        widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax4.plot(range(len(target_vals)), target_vals, 'go-',
                linewidth=2, markersize=8, label='Target', alpha=0.7)
        ax4.set_xticks(range(len(target_vals)))
        ax4.set_xticklabels([f'{t:.1f}' for t in target_vals])
        ax4.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Difficulty Score Distribution', fontsize=12, fontweight='bold')
        ax4.set_title('Distribution of Generated Difficulties', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nEvaluation plot saved to {save_path}")

        plt.show()

        print(f"\n{'='*90}")
        print(f"DETAILED EVALUATION RESULTS")
        print(f"{'='*90}")
        print(f"{'Target':<10} {'Generated':<12} {'Std Dev':<12} {'Error':<12} {'% Error':<12} {'Range':<20}")
        print(f"{'-'*90}")

        for target in target_vals:
            summary = results['summary'][target]
            samples = summary['samples']
            range_str = f"[{min(samples):.3f}, {max(samples):.3f}]"
            pct_error = (summary['error'] / target * 100) if target != 0 else 0
            print(f"{target:<10.2f} {summary['mean_difficulty']:<12.3f} "
                  f"{summary['std_difficulty']:<12.3f} {summary['error']:<12.3f} "
                  f"{pct_error:<12.1f}% {range_str:<20}")

        overall_mae = np.mean([results['summary'][t]['error'] for t in target_vals])
        overall_correlation = np.corrcoef(all_targets, all_actual_scores)[0, 1]
        
        # Calculate accuracy (exact class match)
        correct_predictions = sum(1 for t, a in zip(all_targets, all_actual_scores) if t == a)
        accuracy = correct_predictions / len(all_targets) * 100

        print(f"\n{'='*90}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*90}")
        print(f"Mean Absolute Error (MAE): {overall_mae:.4f}")
        print(f"Correlation Coefficient: {overall_correlation:.4f}")
        print(f"Accuracy (exact class match): {accuracy:.1f}% ({correct_predictions}/{len(all_targets)})")
        print(f"Total Samples Generated: {len(all_targets)}")
        print(f"{'='*90}\n")

        results['overall'] = {
            'mae': overall_mae,
            'correlation': overall_correlation,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_samples': len(all_targets)
        }

        return results
