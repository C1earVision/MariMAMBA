import torch
import torch.nn as nn
import os
import sys

# Add root to path so we can find 'models'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mamba import Mamba

def try_export():
    # 1. Initialize Model (Using common parameters from mamba.py)
    # These should match the training config. Let's guess based on defaults.
    model = Mamba(
        num_tile_types=13,
        column_height=14,
        tile_embed_dim=8,
        d_model=128,
        n_layers=6,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0, # Dropout should be 0 for export
        max_seq_len=32,
        num_attributes=3,
        columns_per_token=1
    )

    # 2. Load weights (if available)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "..", "checkpoints", "mamba_best_ema.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Check if it's a state dict or a wrapped dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Clean state dict (remove 'module.' prefix if it exists)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()

    # 3. Create dummy inputs
    batch_size = 1
    seq_len = 32 # Fixed length for export
    column_sequence = torch.randint(0, 13, (batch_size, seq_len, 14))
    attribute_sequence = torch.randn(batch_size, seq_len, 3)

    print(f"Attempting ONNX export with FIXED seq_len={seq_len}...")
    
    output_file = os.path.join(script_dir, "..", "checkpoints", "mamba_model.onnx")
    
    try:
        # Using the legacy exporter (dynamo=False) for better stability
        torch.onnx.export(
            model,
            (column_sequence, attribute_sequence),
            output_file,
            export_params=True,
            opset_version=18, # Using 18 as requested by the exporter
            do_constant_folding=True,
            input_names=['column_sequence', 'attribute_sequence'],
            output_names=['logits', 'pred_attrs']
        )
        print(f"SUCCESS: Model exported to {output_file}")
    except Exception as e:
        print(f"FAILURE: Export failed.")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    try_export()
