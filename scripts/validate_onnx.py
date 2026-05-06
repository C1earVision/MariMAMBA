import torch
import onnxruntime as ort
import numpy as np
import os
import sys

# Add root to path so we can find 'models'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mamba import Mamba

def validate():
    # 1. Setup Models
    device = "cpu"
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
        dropout=0.0,
        max_seq_len=m_cfg.max_seq_len,
        num_attributes=m_cfg.num_attributes,
        columns_per_token=m_cfg.columns_per_token
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = train_cfg.save_path.replace('.pth', '_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 2. Prepare Inputs
    col_seq = torch.randint(0, m_cfg.num_tile_types, (1, seq_len, m_cfg.column_height))
    attr_seq = torch.randn(1, seq_len, m_cfg.num_attributes)

    # 3. Get PyTorch Output
    with torch.no_grad():
        pt_logits, pt_attrs = model(col_seq, attr_seq)

    # 4. Get ONNX Output
    onnx_path = checkpoint_path.replace('.pth', '.onnx')
    try:
        session = ort.InferenceSession(onnx_path)
        
        ort_inputs = {
            "column_sequence": col_seq.numpy().astype(np.int64),
            "attribute_sequence": attr_seq.numpy().astype(np.float32)
        }
        
        ort_outs = session.run(None, ort_inputs)
        onx_logits, onx_attrs = ort_outs

        # 5. Compare
        logits_diff = np.abs(pt_logits.numpy() - onx_logits).max()
        attrs_diff = np.abs(pt_attrs.numpy() - onx_attrs).max()

        print(f"Comparison Result:")
        print(f"  Logits Max Diff: {logits_diff:.6f}")
        print(f"  Attrs Max Diff:  {attrs_diff:.6f}")
        
        if logits_diff < 1e-4:
            print("VERIFICATION SUCCESS: ONNX matches PyTorch!")
        else:
            print("VERIFICATION FAILURE: Outputs do not match.")
            
    except Exception as e:
        print(f"VERIFICATION ERROR: {str(e)}")

if __name__ == "__main__":
    validate()
