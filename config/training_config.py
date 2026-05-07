from dataclasses import dataclass

@dataclass
class MambaTrainingConfig:
    batch_size: int = 32
    num_epochs: int = 400
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    ema_decay: float = 0.999
    patience: int = 30
    max_seq_len: int = 12
    stride: int = 1
    save_interval: int = 1000
    save_path: str = './checkpoints/mamba.pth'
    use_ema: bool = False

