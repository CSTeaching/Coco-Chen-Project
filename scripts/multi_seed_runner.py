"""Multi-seed training and evaluation runner.

Usage (smoke test):
    python scripts/multi_seed_runner.py --episodes 50 --seeds 42,43,44,45,46

This imports the `train` function from `agents/train_dqn.py` and runs it for each seed,
collects `run_summary` outputs, saves per-seed CSVs and an aggregated summary CSV, and
creates a simple plot of TIR means across seeds.
"""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root in path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.train_dqn import train


def run_multi_seed(seeds: List[int], episodes: int, out_dir: str, **train_kwargs) -> List[Dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for seed in seeds:
        print(f"Running seed={seed} (episodes={episodes})")
        run_out = out_dir / f"dqn_seed_{seed}.csv"

        agent, csv_rows, run_summary = train(
            episodes=episodes,
            seed=seed,
            output_csv=str(run_out),
            eval_freq=train_kwargs.get("eval_freq", 25),
            eval_episodes=train_kwargs.get("eval_episodes", 3),
            insulin_penalty_coeff=train_kwargs.get("insulin_penalty_coeff", 0.1),
            early_stop_patience=train_kwargs.get("early_stop_patience", None),
            early_stop_min_evals=train_kwargs.get("early_stop_min_evals", 4),
            safety_hypo_max=train_kwargs.get("safety_hypo_max", 3.0),
            safety_severe_max=train_kwargs.get("safety_severe_max", 2.0),
            lr=train_kwargs.get("lr", 0.001),
            gamma=train_kwargs.get("gamma", 0.99),
            replay_size=train_kwargs.get("replay_size", 10000),
            batch_size=train_kwargs.get("batch_size", 64),
            target_update_freq=train_kwargs.get("target_update_freq", 100),
            epsilon_start=train_kwargs.get("epsilon_start", 1.0),
            epsilon_end=train_kwargs.get("epsilon_end", 0.1),
            epsilon_decay=train_kwargs.get("epsilon_decay", 500),
            verbose=False,
        )

        summary = run_summary or {}
        summary["seed"] = seed
        summary["csv"] = str(run_out)
        summaries.append(summary)

    # Save aggregated summary
    summary_df = pd.DataFrame(summaries)
    summary_csv = out_dir / "multi_seed_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Plot TIR across seeds if available
    if "eval_tir_mean" in summary_df.columns:
        plt.figure(figsize=(6,4))
        plt.plot(summary_df['seed'], summary_df['eval_tir_mean'], marker='o')
        plt.xlabel('Seed')
        plt.ylabel('Eval TIR Mean (%)')
        plt.title('Multi-seed Eval TIR')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / 'multi_seed_tir.png', dpi=160)
        plt.close()

    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=str, default="artifacts/multi_seed")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summaries = run_multi_seed(seeds, args.episodes, args.out)
    print("Run complete. Summaries saved to", args.out)


if __name__ == '__main__':
    main()
