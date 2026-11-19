"""Generate a Hugging Face Hub model card (README.md) for a trained run."""

from __future__ import annotations

from typing import Dict, Optional

from .config import TrainConfig


def render_model_card(
    cfg: TrainConfig,
    metrics: Dict[str, float],
    baseline: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, float]] = None,
) -> str:
    lines = [
        "---",
        "library_name: peft",
        f"base_model: {cfg.model_name}",
        f"datasets:\n- {cfg.dataset_name}",
        "pipeline_tag: text-classification",
        "license: mit",
        "tags:\n- lora\n- text-classification\n- tinytune",
        "---",
        "",
        f"# TinyTune: {cfg.model_name} + LoRA on {cfg.dataset_name}",
        "",
        "A LoRA adapter fine-tuned on a single free GPU with "
        "[TinyTune](https://github.com/techdeeplimbachiya/tinytune).",
        "",
        "## Results",
        "",
        "| Model | Accuracy | F1 (macro) |",
        "| --- | --- | --- |",
    ]
    if baseline:
        lines.append(f"| Base model (no fine-tuning) | {baseline['accuracy']:.4f} | {baseline['f1']:.4f} |")
    lines.append(f"| TinyTune LoRA | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} |")
    lines += [
        "",
        f"Evaluated on {cfg.eval_samples} held-out examples.",
        "",
        "## Training setup",
        "",
        f"- Train examples: {cfg.train_samples}, epochs: {cfg.epochs}, batch size: {cfg.batch_size}",
        f"- Learning rate: {cfg.learning_rate}, max length: {cfg.max_length}",
        f"- LoRA r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}, "
        f"targets={', '.join(cfg.lora_target_modules)}",
    ]
    if params:
        lines.append(
            f"- Trainable parameters: {int(params['trainable']):,} of {int(params['total']):,} "
            f"({params['trainable_pct']}%)"
        )
    lines += [
        "",
        "## Usage",
        "",
        "```python",
        "from peft import AutoPeftModelForSequenceClassification",
        "from transformers import AutoTokenizer",
        "",
        f'repo = "{cfg.hub_repo_id or cfg.output_dir}"',
        "model = AutoPeftModelForSequenceClassification.from_pretrained(repo)",
        "tokenizer = AutoTokenizer.from_pretrained(repo)",
        "```",
        "",
    ]
    return "\n".join(lines)
