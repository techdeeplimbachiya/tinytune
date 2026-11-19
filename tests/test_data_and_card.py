from tinytune.config import TrainConfig
from tinytune.data import tokenize_batch
from tinytune.model_card import render_model_card


def fake_tokenizer(texts, truncation=True, max_length=8):
    ids = [[len(word) for word in t.split()][:max_length] for t in texts]
    return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


def test_tokenize_batch_renames_labels_and_truncates():
    batch = {"text": ["a bb ccc dddd", "hello world"], "label": [1, 0]}
    out = tokenize_batch(batch, fake_tokenizer, "text", "label", max_length=3)
    assert out["labels"] == [1, 0]
    assert out["input_ids"][0] == [1, 2, 3]
    assert "text" not in out


def test_model_card_contains_results_and_baseline():
    cfg = TrainConfig(hub_repo_id="user/tinytune")
    card = render_model_card(
        cfg,
        metrics={"accuracy": 0.89, "f1": 0.88},
        baseline={"accuracy": 0.5, "f1": 0.33},
        params={"total": 1000, "trainable": 10, "trainable_pct": 1.0},
    )
    assert card.startswith("---")
    assert "| TinyTune LoRA | 0.8900 | 0.8800 |" in card
    assert "Base model (no fine-tuning)" in card
    assert 'repo = "user/tinytune"' in card
