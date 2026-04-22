from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import sacrebleu


def compute_bleu(preds: List[str], refs: List[str]) -> float:
    score = sacrebleu.corpus_bleu(preds, [refs])
    return float(score.score)


def save_results(bleu_baseline: float, bleu_mt5: float, out_csv: str = "artifacts/results.csv") -> pd.DataFrame:
    delta = bleu_mt5 - bleu_baseline
    df = pd.DataFrame(
        [
            {"model": "baseline_lstm", "bleu": bleu_baseline},
            {"model": "mt5_small", "bleu": bleu_mt5},
            {"model": "improvement_delta", "bleu": delta},
        ]
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def plot_results(df: pd.DataFrame, out_png: str = "artifacts/bleu_comparison.png") -> str:
    chart_df = df[df["model"].isin(["baseline_lstm", "mt5_small"])]
    plt.figure(figsize=(6, 4))
    plt.bar(chart_df["model"], chart_df["bleu"], color=["steelblue", "seagreen"])
    plt.title("BLEU Comparison: Baseline vs mT5")
    plt.ylabel("BLEU")
    plt.tight_layout()

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return str(out_path)
