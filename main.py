import argparse

import torch

from baseline import predict_baseline, train_baseline
from data import build_baseline_vocabs, load_parallel_data
from evaluate import compute_bleu, plot_results, save_results
from mt5 import predict_mt5, train_mt5


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline LSTM vs mT5 for EN->TE translation")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--baseline-epochs", type=int, default=3)
    parser.add_argument("--mt5-epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading EN-TE parallel dataset...")
    dataset = load_parallel_data(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    print("Building baseline vocab...")
    src_vocab, tgt_vocab = build_baseline_vocabs(dataset["train"])

    print("Training baseline LSTM...")
    baseline_artifacts = train_baseline(
        dataset["train"],
        dataset["validation"],
        src_vocab,
        tgt_vocab,
        epochs=args.baseline_epochs,
        device=args.device,
    )

    print("Training mT5...")
    mt5_model, mt5_tokenizer = train_mt5(
        dataset,
        epochs=args.mt5_epochs,
    )

    src_test = dataset["test"]["en"]
    refs_test = dataset["test"]["te"]

    print("Generating baseline predictions...")
    pred_baseline = predict_baseline(baseline_artifacts, src_test, device=args.device)

    print("Generating mT5 predictions...")
    pred_mt5 = predict_mt5(mt5_model, mt5_tokenizer, src_test, device=args.device)

    bleu_baseline = compute_bleu(pred_baseline, refs_test)
    bleu_mt5 = compute_bleu(pred_mt5, refs_test)
    delta = bleu_mt5 - bleu_baseline

    print("\n=== BLEU Comparison ===")
    print(f"BLEU_baseline: {bleu_baseline:.2f}")
    print(f"BLEU_mT5:      {bleu_mt5:.2f}")
    print(f"Improvement:   {delta:+.2f}")

    df = save_results(bleu_baseline, bleu_mt5)
    chart_path = plot_results(df)

    print(f"\nSaved results CSV: artifacts/results.csv")
    print(f"Saved chart PNG:   {chart_path}")


if __name__ == "__main__":
    main()
