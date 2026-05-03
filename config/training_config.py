from dataclasses import dataclass

@dataclass
class MambaTrainingConfig:
    batch_size: int = 32
    num_epochs: int = 300
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    ema_decay: float = 0.999
    patience: int = 30
    max_seq_len: int = 32
    stride: int = 8
    save_interval: int = 100
    save_path: str = './checkpoints/mamba.pth'

