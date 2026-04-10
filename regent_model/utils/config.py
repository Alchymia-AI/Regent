"""Configuration loading utilities."""

from dataclasses import dataclass

import yaml


@dataclass
class TrainConfig:
    max_seq_len: int = 2048
    batch_size: int = 4
    gradient_accumulation: int = 4
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    max_steps: int = 100000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dtype: str = "float16"

    # Ver head training (phase 3)
    ver_lr: float = 1e-4
    ver_freeze_backbone: bool = True
    ver_epochs: int = 5

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        t = raw.get("training", {})
        ver = t.get("ver_head", {})

        return cls(
            max_seq_len=t.get("max_seq_len", 2048),
            batch_size=t.get("batch_size", 4),
            gradient_accumulation=t.get("gradient_accumulation", 4),
            lr=t.get("lr", 3e-4),
            min_lr=t.get("min_lr", 3e-5),
            warmup_steps=t.get("warmup_steps", 1000),
            max_steps=t.get("max_steps", 100000),
            weight_decay=t.get("weight_decay", 0.1),
            grad_clip=t.get("grad_clip", 1.0),
            dtype=t.get("dtype", "float16"),
            ver_lr=ver.get("lr", 1e-4),
            ver_freeze_backbone=ver.get("freeze_backbone", True),
            ver_epochs=ver.get("epochs", 5),
        )
