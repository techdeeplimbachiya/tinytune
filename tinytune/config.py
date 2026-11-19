"""Training configuration for TinyTune."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    """All settings for one fine-tuning run.

    Defaults are chosen so a run finishes in roughly 10-15 minutes on a free
    Google Colab T4 GPU.
    """

    # Model and data
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "imdb"
    text_column: str = "text"
    label_column: str = "label"
    num_labels: int = 2
    label_names: List[str] = field(default_factory=lambda: ["negative", "positive"])
    train_samples: int = 4000
    eval_samples: int = 1000
    max_length: int = 256

    # LoRA (PEFT)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_lin", "v_lin"])
    modules_to_save: List[str] = field(default_factory=lambda: ["pre_classifier", "classifier"])

    # Optimization
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    fp16: bool = True
    seed: int = 42

    # Output
    output_dir: str = "outputs/tinytune-distilbert-imdb"
    push_to_hub: bool = False
    hub_repo_id: Optional[str] = None

    def validate(self) -> None:
        """Raise ValueError if the configuration is inconsistent."""
        if self.num_labels != len(self.label_names):
            raise ValueError(
                f"num_labels ({self.num_labels}) must match label_names ({len(self.label_names)})"
            )
        if self.train_samples <= 0 or self.eval_samples <= 0:
            raise ValueError("train_samples and eval_samples must be positive")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.push_to_hub and not self.hub_repo_id:
            raise ValueError("hub_repo_id is required when push_to_hub is True")

    def to_dict(self) -> dict:
        return asdict(self)
