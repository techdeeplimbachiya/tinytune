"""Gradio demo. Runs locally or as a Hugging Face Space.

Set MODEL_PATH to a local output folder or a Hub repo id, e.g.
    MODEL_PATH=your-username/tinytune-distilbert-imdb python app.py
"""

import json
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/tinytune-distilbert-imdb")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoPeftModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def load_labels(path: str, num_labels: int) -> list:
    """Read the label names saved by training (labels.json), locally or from the Hub."""
    try:
        local = os.path.join(path, "labels.json")
        file = local if os.path.exists(local) else hf_hub_download(path, "labels.json")
        with open(file, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return [f"LABEL_{i}" for i in range(num_labels)]


labels = load_labels(MODEL_PATH, model.config.num_labels)


def classify(text: str) -> dict:
    if not text.strip():
        return {}
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(dim=-1)[0].tolist()
    return {label: float(p) for label, p in zip(labels, probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=4, placeholder="Type a movie review..."),
    outputs=gr.Label(num_top_classes=2),
    title="TinyTune: GPU-poor sentiment classifier",
    description="DistilBERT + LoRA, fine-tuned on IMDb with a single free GPU.",
    examples=[
        ["An absolute masterpiece. I was smiling the whole time."],
        ["Two hours I will never get back. The plot made no sense."],
    ],
)

if __name__ == "__main__":
    demo.launch()
