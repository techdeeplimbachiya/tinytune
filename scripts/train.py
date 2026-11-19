"""Command-line entry point: python scripts/train.py --epochs 2 --train-samples 4000"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tinytune.config import TrainConfig  # noqa: E402


def parse_args(argv=None) -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Fine-tune a small transformer with LoRA.")
    p.add_argument("--model-name", default=defaults.model_name)
    p.add_argument("--dataset-name", default=defaults.dataset_name)
    p.add_argument("--train-samples", type=int, default=defaults.train_samples)
    p.add_argument("--eval-samples", type=int, default=defaults.eval_samples)
    p.add_argument("--max-length", type=int, default=defaults.max_length)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    p.add_argument("--lora-r", type=int, default=defaults.lora_r)
    p.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    p.add_argument("--no-fp16", action="store_true", help="Disable mixed precision")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--output-dir", default=defaults.output_dir)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo-id", default=None, help="e.g. your-username/tinytune-distilbert-imdb")
    a = p.parse_args(argv)
    cfg = TrainConfig(
        model_name=a.model_name,
        dataset_name=a.dataset_name,
        train_samples=a.train_samples,
        eval_samples=a.eval_samples,
        max_length=a.max_length,
        epochs=a.epochs,
        batch_size=a.batch_size,
        learning_rate=a.learning_rate,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        fp16=not a.no_fp16,
        seed=a.seed,
        output_dir=a.output_dir,
        push_to_hub=a.push_to_hub,
        hub_repo_id=a.hub_repo_id,
    )
    cfg.validate()
    return cfg


def main(argv=None) -> None:
    from tinytune.train import train

    results = train(parse_args(argv))
    print(f"Baseline: {results['baseline']}")
    print(f"Fine-tuned: {results['final']}")


if __name__ == "__main__":
    main()
