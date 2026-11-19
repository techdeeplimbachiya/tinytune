"""Building a LoRA-wrapped sequence-classification model with PEFT."""

from __future__ import annotations

from .config import TrainConfig


def build_model(cfg: TrainConfig):
    """Load the base model and wrap it with LoRA adapters.

    Only the LoRA matrices and the classification head are trained, which keeps
    memory low enough for a single free GPU.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    id2label = {i: name for i, name in enumerate(cfg.label_names)}
    label2id = {name: i for i, name in id2label.items()}
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels, id2label=id2label, label2id=label2id
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        modules_to_save=cfg.modules_to_save,
    )
    return get_peft_model(base, lora)


def count_parameters(model) -> dict:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(100.0 * trainable / total, 3) if total else 0.0,
    }
