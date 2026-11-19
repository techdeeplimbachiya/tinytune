# TinyTune

**Fine-tune a transformer on a single free GPU.** TinyTune fine-tunes DistilBERT for text classification with LoRA (PEFT), using Hugging Face Transformers, Datasets and Accelerate. Only about 1% of the model's weights are trained, so a full run fits on a free Google Colab T4 GPU.

[![CI](https://github.com/techdeeplimbachiya/tinytune/actions/workflows/ci.yml/badge.svg)](https://github.com/techdeeplimbachiya/tinytune/actions/workflows/ci.yml)

## Results

| Model | Accuracy | F1 (macro) |
| --- | --- | --- |
| DistilBERT, no fine-tuning (baseline) | 0.408 | 0.406 |
| DistilBERT + LoRA (TinyTune) | **0.869** | **0.869** |

IMDb, 4,000 training reviews, 1,000 held-out test reviews, 2 epochs on one free Colab T4 GPU. Trainable parameters: 739,586 of 67,694,596 (1.09%).

- Model on the Hub: [techdeepml/tinytune-distilbert-imdb](https://huggingface.co/techdeepml/tinytune-distilbert-imdb)
- Live demo (Gradio Space): _coming soon_

## How it works

| File | What it does |
| --- | --- |
| `tinytune/config.py` | One dataclass holding every setting, with validation |
| `tinytune/data.py` | Loads IMDb with `datasets`, subsamples, tokenizes |
| `tinytune/model.py` | Wraps DistilBERT with LoRA adapters on the attention `q_lin` / `v_lin` layers |
| `tinytune/train.py` | Accelerate training loop: same code on CPU, one GPU or several; evaluates the baseline before training |
| `tinytune/metrics.py` | Accuracy and macro F1 in plain NumPy |
| `tinytune/model_card.py` | Writes the Hub model card with the results table |
| `app.py` | Gradio demo, runs locally or as a Hugging Face Space |
| `tests/` | pytest unit tests, run on every push by GitHub Actions |

## Quick start (Google Colab, free GPU)

Open `notebooks/train_colab.ipynb` in Colab, choose **Runtime → Change runtime type → T4 GPU**, and run the cells. Or run the commands yourself:

```bash
git clone https://github.com/techdeeplimbachiya/tinytune.git
cd tinytune
pip install -r requirements.txt
python scripts/train.py --epochs 2 --train-samples 4000 --eval-samples 1000
```

The run prints the baseline and fine-tuned scores, then saves the adapter, tokenizer, `results.json` and a model card to `outputs/tinytune-distilbert-imdb/`.

### Push to the Hugging Face Hub

```bash
huggingface-cli login   # paste a write token from huggingface.co/settings/tokens
python scripts/train.py --push-to-hub --hub-repo-id <your-username>/tinytune-distilbert-imdb
```

### Run the demo

```bash
MODEL_PATH=<your-username>/tinytune-distilbert-imdb python app.py
```

To host it as a Space: create a Gradio Space, upload `app.py` and `requirements.txt`, and add a Space variable `MODEL_PATH` set to your Hub repo id.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Options

```bash
python scripts/train.py --help
```

Useful flags: `--model-name`, `--dataset-name`, `--train-samples`, `--epochs`, `--batch-size`, `--learning-rate`, `--lora-r`, `--lora-alpha`, `--no-fp16`.

## License

MIT
