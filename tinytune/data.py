"""Loading and tokenizing datasets with Hugging Face Datasets."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from .config import TrainConfig


def tokenize_batch(
    batch: Dict[str, List[Any]],
    tokenizer: Callable[..., Dict[str, List[Any]]],
    text_column: str,
    label_column: str,
    max_length: int,
) -> Dict[str, List[Any]]:
    """Tokenize one batch of examples and rename the label column to ``labels``.

    Kept free of heavy imports so it can be unit-tested with a fake tokenizer.
    """
    encoded = tokenizer(batch[text_column], truncation=True, max_length=max_length)
    encoded = dict(encoded)
    encoded["labels"] = list(batch[label_column])
    return encoded


def load_splits(cfg: TrainConfig, tokenizer) -> Tuple[Any, Any]:
    """Return tokenized (train, eval) datasets, subsampled for a fast run."""
    from datasets import load_dataset

    raw = load_dataset(cfg.dataset_name)
    train = raw["train"].shuffle(seed=cfg.seed).select(range(min(cfg.train_samples, len(raw["train"]))))
    eval_split = "test" if "test" in raw else "validation"
    evaluation = raw[eval_split].shuffle(seed=cfg.seed).select(
        range(min(cfg.eval_samples, len(raw[eval_split])))
    )

    def _tok(batch):
        return tokenize_batch(batch, tokenizer, cfg.text_column, cfg.label_column, cfg.max_length)

    columns_to_drop = train.column_names
    train = train.map(_tok, batched=True, remove_columns=columns_to_drop)
    evaluation = evaluation.map(_tok, batched=True, remove_columns=columns_to_drop)
    return train, evaluation
