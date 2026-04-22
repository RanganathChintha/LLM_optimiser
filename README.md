# EN→TE Machine Translation Comparison

This project compares English→Telugu translation performance between:

1. **Baseline**: a lightweight word-level Seq2Seq LSTM.
2. **Transformer**: fine-tuned `google/mt5-small` via HuggingFace Transformers.

It uses a small split from the OPUS100 English-Telugu parallel corpus, evaluates both models with sacreBLEU, and saves results to CSV + chart.

## Project structure

- `data.py` - dataset loading, cleaning, tokenization, vocab
- `baseline.py` - baseline seq2seq LSTM model and training/prediction
- `mt5.py` - mT5 fine-tuning and prediction
- `evaluate.py` - BLEU scoring, CSV output, and chart plotting
- `main.py` - end-to-end training and comparison pipeline
- `app.py` - optional FastAPI endpoint for both models

## Install

```bash
pip install -r requirements.txt
```

## Run experiment

```bash
python main.py --train-size 1000 --val-size 200 --test-size 200 --baseline-epochs 3 --mt5-epochs 1
```

Expected output includes:
- `BLEU_baseline`
- `BLEU_mT5`
- `Improvement`
- `artifacts/results.csv`
- `artifacts/bleu_comparison.png`

## Optional API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /translate` with JSON body: `{"text": "How are you?"}`

> Note: startup triggers quick model training for demo use.
