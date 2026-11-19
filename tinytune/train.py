"""Training loop built on Hugging Face Accelerate (same code on CPU or GPU)."""

from __future__ import annotations

import json
import math
import os
from typing import Dict

from .config import TrainConfig
from .metrics import compute_metrics
from .model_card import render_model_card


def evaluate(model, dataloader, accelerator) -> Dict[str, float]:
    import torch

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            preds, labels = accelerator.gather_for_metrics((preds, batch["labels"]))
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return compute_metrics(all_preds, all_labels)


def train(cfg: TrainConfig) -> Dict[str, Dict[str, float]]:
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

    from .data import load_splits
    from .model import build_model, count_parameters

    cfg.validate()
    use_fp16 = cfg.fp16 and torch.cuda.is_available()
    accelerator = Accelerator(mixed_precision="fp16" if use_fp16 else "no")
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds, eval_ds = load_splits(cfg, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    eval_dl = DataLoader(eval_ds, batch_size=cfg.batch_size * 2, collate_fn=collator)

    model = build_model(cfg)
    params = count_parameters(model)
    accelerator.print(
        f"Trainable params: {params['trainable']:,} / {params['total']:,} ({params['trainable_pct']}%)"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.epochs * math.ceil(len(train_dl))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps
    )
    model, optimizer, train_dl, eval_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, eval_dl, scheduler
    )

    baseline = evaluate(model, eval_dl, accelerator)
    accelerator.print(f"Baseline (before training): {baseline}")

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl, start=1):
            loss = model(**batch).loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                accelerator.print(f"epoch {epoch + 1} step {step}/{len(train_dl)} loss {running / step:.4f}")
        metrics = evaluate(model, eval_dl, accelerator)
        accelerator.print(f"Epoch {epoch + 1} eval: {metrics}")

    results = {"baseline": baseline, "final": metrics}
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        card = render_model_card(cfg, metrics, baseline, params)
        with open(os.path.join(cfg.output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(card)
        with open(os.path.join(cfg.output_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"config": cfg.to_dict(), "params": params, **results}, f, indent=2)
        with open(os.path.join(cfg.output_dir, "labels.json"), "w", encoding="utf-8") as f:
            json.dump(cfg.label_names, f)
        if cfg.push_to_hub:
            from huggingface_hub import create_repo, upload_folder

            create_repo(cfg.hub_repo_id, exist_ok=True)
            upload_folder(repo_id=cfg.hub_repo_id, folder_path=cfg.output_dir)
            accelerator.print(f"Pushed to https://huggingface.co/{cfg.hub_repo_id}")
        accelerator.print(f"Saved to {cfg.output_dir}")
    return results
