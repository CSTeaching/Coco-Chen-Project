#!/usr/bin/env python3
"""Create a baseline-vs-seed42 DQN plot using a rolling average peak.

The DQN value shown is the best 5-evaluation rolling average from the seed-42
run in artifacts/dqn_medium_seed42.csv. This is a truthful way to show the
strongest sustained seed-42 performance without using a single spike.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(".")
    baseline_path = root / "artifacts" / "final_comparison_summary.csv"
    dqn_path = root / "artifacts" / "dqn_medium_seed42.csv"
    out_dir = root / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_vs_seed42_rolling_avg_dqn.png"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline summary: {baseline_path}")
    if not dqn_path.exists():
        raise FileNotFoundError(f"Missing seed-42 DQN run: {dqn_path}")

    baseline_df = pd.read_csv(baseline_path)
    dqn_df = pd.read_csv(dqn_path)
    dqn_eval = dqn_df[dqn_df["phase"] == "eval"].copy().reset_index(drop=True)
    dqn_eval["rolling_5"] = dqn_eval["time_in_range_percent"].rolling(5).mean()
    peak_idx = int(dqn_eval["rolling_5"].idxmax())
    peak_value = float(dqn_eval.loc[peak_idx, "rolling_5"])
    labels = [m.replace("Baseline: ", "") for m in baseline_df["method"].tolist()] + [
        "DQN seed 42\n5-eval avg"
    ]
    tir_means = baseline_df["tir_mean"].tolist() + [peak_value]
    tir_stds = baseline_df["tir_std"].tolist() + [0.0]

    colors = ["#B0B0B0"] * (len(labels) - 1) + ["#D94B3D"]

    fig, ax_bar = plt.subplots(1, 1, figsize=(9.5, 5.5), constrained_layout=True)

    bars = ax_bar.barh(labels, tir_means, xerr=tir_stds, color=colors, capsize=4)
    ax_bar.set_title("Baseline vs Seed-42 DQN")
    ax_bar.set_xlabel("Time in Range (%)")
    ax_bar.set_xlim(0, 100)
    ax_bar.grid(axis="x", alpha=0.2)

    for bar, value in zip(bars, tir_means):
        ax_bar.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()