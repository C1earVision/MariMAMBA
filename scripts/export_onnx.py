import torch
import torch.nn as nn
import os
import sys

# Add root to path so we can find 'models'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mamba import Mamba

def try_export():
    from config.model_config import MambaConfig
    m_cfg = MambaConfig()
    seq_len = m_cfg.max_seq_len 

    from config.training_config import MambaTrainingConfig
    train_cfg = MambaTrainingConfig()

    model = Mamba(
        num_tile_types=m_cfg.num_tile_types,
        column_height=m_cfg.column_height,
        tile_embed_dim=m_cfg.tile_embed_dim,
        d_model=m_cfg.d_model,
        n_layers=m_cfg.n_layers,
        d_state=m_cfg.d_state,
        d_conv=m_cfg.d_conv,
        expand=m_cfg.expand,
        dropout=0.0, # Dropout should be 0 for export
        max_seq_len=m_cfg.max_seq_len,
        num_attributes=m_cfg.num_attributes,
        columns_per_token=m_cfg.columns_per_token
    )

    # 2. Load weights (if available)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Use the 'best' checkpoint path automatically
    checkpoint_path = train_cfg.save_path.replace('.pth', '_best.pth')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Check if it's a state dict or a wrapped dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Clean state dict (remove 'module.' prefix if it exists)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Exporting random weights.")
    
    model.eval()

    # 3. Create dummy inputs
    batch_size = 1
    seq_len = m_cfg.max_seq_len # Use configured max length for export
    column_sequence = torch.randint(0, m_cfg.num_tile_types, (batch_size, seq_len, m_cfg.column_height))
    attribute_sequence = torch.randn(batch_size, seq_len, m_cfg.num_attributes)

    print(f"Attempting ONNX export with seq_len={seq_len}...")
    
    output_file = checkpoint_path.replace('.pth', '.onnx')
    
    try:
        # Using the legacy exporter (dynamo=False) for better stability
        torch.onnx.export(
            model,
            (column_sequence, attribute_sequence),
            output_file,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
            input_names=['column_sequence', 'attribute_sequence'],
            output_names=['logits', 'pred_attrs'],
            dynamic_axes={
                'column_sequence': {0: 'batch_size', 1: 'sequence_length'},
                'attribute_sequence': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'},
                'pred_attrs': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        print(f"SUCCESS: Model exported to {output_file}")
    except Exception as e:
        print(f"FAILURE: Export failed.")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    try_export()
