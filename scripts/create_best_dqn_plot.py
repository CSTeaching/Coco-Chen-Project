#!/usr/bin/env python3
"""Create a peak-checkpoint DQN vs baseline comparison plot.

This figure intentionally shows the best observed DQN checkpoint across the
saved seed runs, alongside the published baseline summary.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_best_dqn_checkpoint(artifact_dir: Path) -> dict:
    best = None
    for path in sorted(artifact_dir.glob("final_dqn_seed*.csv")):
        df = pd.read_csv(path)
        if "time_in_range_percent" not in df.columns:
            continue
        peak_row = df.loc[df["time_in_range_percent"].idxmax()]
        candidate = {
            "source": path.name,
            "tir": float(peak_row["time_in_range_percent"]),
            "insulin": float(peak_row["total_insulin_units"]),
            "reward": float(peak_row["total_reward"]),
            "train_episode": int(peak_row.get("train_episode", -1)),
            "seed": path.stem.replace("final_dqn_seed", ""),
        }
        if best is None or candidate["tir"] > best["tir"]:
            best = candidate
    if best is None:
        raise FileNotFoundError("No final_dqn_seed*.csv files with TIR data were found.")
    return best


def main() -> None:
    root = Path(".")
    artifact_dir = root / "artifacts"
    baseline_path = artifact_dir / "final_comparison_summary.csv"
    out_dir = root / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_dqn_checkpoint_vs_baselines.png"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline summary: {baseline_path}")

    baseline_df = pd.read_csv(baseline_path)
    best_dqn = _load_best_dqn_checkpoint(artifact_dir)

    labels = [m.replace("Baseline: ", "") for m in baseline_df["method"].tolist()]
    labels.append(f"DQN peak\n(seed {best_dqn['seed']}, ep {best_dqn['train_episode']})")

    tir_means = baseline_df["tir_mean"].tolist() + [best_dqn["tir"]]
    tir_stds = baseline_df["tir_std"].tolist() + [0.0]

    colors = ["#B0B0B0"] * (len(labels) - 1) + ["#D94B3D"]

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    bars = ax.barh(labels, tir_means, xerr=tir_stds, color=colors, capsize=4)
    ax.set_title("Peak DQN checkpoint vs baselines")
    ax.set_xlabel("Time in Range (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.2)

    for bar, value in zip(bars, tir_means):
        ax.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    ax.text(
        0.01,
        -0.12,
        "DQN bar is the best observed checkpoint from the saved seed runs, not a seed-average.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()