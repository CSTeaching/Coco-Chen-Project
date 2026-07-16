#!/usr/bin/env python3
"""Create a fair baseline-vs-DQN comparison plot.

This uses the published aggregated baseline summary and the seed-averaged DQN
metrics already saved in artifacts/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _short_label(method: str) -> str:
    if method.startswith("Baseline: "):
        return method.replace("Baseline: ", "")
    return method


def main() -> None:
    root = Path(".")
    baseline_path = root / "artifacts" / "final_comparison_summary.csv"
    dqn_path = root / "artifacts" / "seed_averaged_metrics.csv"
    out_dir = root / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_vs_dqn_tir_insulin.png"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline summary: {baseline_path}")
    if not dqn_path.exists():
        raise FileNotFoundError(f"Missing DQN summary: {dqn_path}")

    baseline_df = pd.read_csv(baseline_path)
    dqn_df = pd.read_csv(dqn_path)

    dqn_lookup = {
        row["metric"]: row for _, row in dqn_df.iterrows()
    }

    labels = [_short_label(method) for method in baseline_df["method"].tolist()] + ["DQN (seed-avg)"]
    tir_means = baseline_df["tir_mean"].tolist() + [float(dqn_lookup["time_in_range_percent"]["mean_across_seeds"])]
    tir_stds = baseline_df["tir_std"].tolist() + [float(dqn_lookup["time_in_range_percent"]["std_across_seeds"])]
    insulin_means = baseline_df["insulin_mean"].tolist() + [float(dqn_lookup["total_insulin_units"]["mean_across_seeds"])]
    insulin_stds = baseline_df["insulin_std"].tolist() + [float(dqn_lookup["total_insulin_units"]["std_across_seeds"])]

    colors = ["#A0A0A0", "#B5B5B5", "#C7C7C7", "#D94B3D"]

    fig, (ax_tir, ax_insulin) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    bars_tir = ax_tir.barh(labels, tir_means, xerr=tir_stds, color=colors, capsize=4)
    ax_tir.set_title("Time in Range (Mean ± Std)")
    ax_tir.set_xlabel("TIR (%)")
    ax_tir.set_xlim(0, 100)
    ax_tir.grid(axis="x", alpha=0.2)

    bars_insulin = ax_insulin.barh(labels, insulin_means, xerr=insulin_stds, color=colors, capsize=4)
    ax_insulin.set_title("Total Insulin (Mean ± Std)")
    ax_insulin.set_xlabel("Insulin (U / episode)")
    ax_insulin.grid(axis="x", alpha=0.2)

    # Highlight the DQN bar visually, but keep the scale and numbers honest.
    bars_tir[-1].set_color("#D94B3D")
    bars_insulin[-1].set_color("#D94B3D")

    for ax, means in ((ax_tir, tir_means), (ax_insulin, insulin_means)):
        for bar, value in zip(ax.patches, means):
            ax.text(
                bar.get_width() + (1.2 if ax is ax_tir else max(means) * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#333333",
            )

    fig.suptitle("Baseline vs DQN: seed-averaged comparison", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()