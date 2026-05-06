from dataclasses import dataclass, field
@dataclass
class MambaConfig:
    num_tile_types: int = 13
    column_height: int = 14
    tile_embed_dim: int = 8
    d_model: int = 128
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.2
    max_seq_len: int = 8
    columns_per_token: int = 1
    num_attributes: int = 3
