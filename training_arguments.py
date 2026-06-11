from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class PR2MTrainingArguments(TrainingArguments):
    """Custom training arguments for BiPRM."""

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Training data path or HF dataset name (e.g. trl-lib/math_shepherd)."},
    )

    loss_type: str = field(
        default="bce",
        metadata={"help": "Loss function type: 'bce' | 'mse' | 'rank'."},
    )

    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-Math-1.5B",
        metadata={"help": "Path to pretrained model or HF model identifier."},
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to allow custom modeling code from the Hub. Set to True only "
                "for repositories you trust."
            )
        },
    )
