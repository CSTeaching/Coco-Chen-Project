#!/usr/bin/env python3
"""Generate final comparison plots from multi-seed results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_final_results() -> None:
    """Generate comparison plots."""
    out_dir = Path("plots/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Read comparison summary
    comp_path = Path("artifacts/final_comparison_summary.csv")
    if not comp_path.exists():
        print(f"Error: {comp_path} not found. Run final_experiments.py first.")
        return
    
    df = pd.read_csv(comp_path)
    
    # Extract numeric values
    methods = df["method"].tolist()
    tir_means = [float(x) for x in df["tir_mean"]]
    tir_stds = [float(x) for x in df["tir_std"]]
    insulin_means = [float(x) for x in df["insulin_mean"]]
    insulin_stds = [float(x) for x in df["insulin_std"]]
    reward_means = [float(x) for x in df["reward_mean"]]
    reward_stds = [float(x) for x in df["reward_std"]]
    
    # Plot 1: TIR and Insulin Tradeoff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ["#E74C3C" if "DQN" in m else "#3498DB" for m in methods]
    ax1.barh(methods, tir_means, xerr=tir_stds, color=colors, capsize=5, alpha=0.8)
    ax1.set_xlabel("Time in Range (%)")
    ax1.set_title("TIR Comparison (Mean ± Std)")
    ax1.set_xlim(0, 100)
    
    ax2.barh(methods, insulin_means, xerr=insulin_stds, color=colors, capsize=5, alpha=0.8)
    ax2.set_xlabel("Total Insulin (U/episode)")
    ax2.set_title("Insulin Usage (Mean ± Std)")
    
    fig.tight_layout()
    fig.savefig(out_dir / "01_tir_insulin_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_dir / '01_tir_insulin_comparison.png'}")
    
    # Plot 2: Reward Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(methods, reward_means, xerr=reward_stds, color=colors, capsize=5, alpha=0.8)
    ax.set_xlabel("Total Reward (Mean ± Std)")
    ax.set_title("Reward Comparison Across Methods")
    fig.tight_layout()
    fig.savefig(out_dir / "02_reward_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_dir / '02_reward_comparison.png'}")
    
    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    plot_final_results()
