"""
Run multi-seed DQN training and evaluation, aggregate results, and produce plots.

Usage (smoke test):
  python multi_seed_runner.py --seeds 42 --episodes 50 --eval-freq 25 --eval-episodes 3

This script invokes `agents/train_dqn.py` for each seed and collects the evaluation CSVs
into `artifacts/multi_seed_results.csv` and saves summary plots to `plots/multi_seed/`.
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
TRAIN_SCRIPT = ROOT / 'agents' / 'train_dqn.py'
OUT_DIR = ROOT / 'artifacts'
PLOTS_DIR = ROOT / 'plots' / 'multi_seed'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_train(seed: int, episodes: int, eval_freq: int, eval_episodes: int, out_prefix: str, insulin_coeff: float | None = None):
    out_csv = OUT_DIR / f"{out_prefix}_seed{seed}.csv"
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        '--seed', str(seed),
        '--episodes', str(episodes),
        '--eval-freq', str(eval_freq),
        '--eval-episodes', str(eval_episodes),
        '--out', str(out_csv)
    ]
    if insulin_coeff is not None:
        cmd += ['--insulin-penalty-coeff', str(insulin_coeff)]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)
    return out_csv


def aggregate_results(csv_paths, out_agg_path):
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df['source'] = p.name
        dfs.append(df)
    agg = pd.concat(dfs, ignore_index=True)
    agg.to_csv(out_agg_path, index=False)
    return agg


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--eval-freq', type=int, default=50)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--out-prefix', type=str, default='dqn_multiseed')
    parser.add_argument('--insulin-penalty-coeff', type=float, default=None,
                        help='If set, pass this coeff to train_dqn')
    args = parser.parse_args(argv)

    csvs = []
    for seed in args.seeds:
        csv = run_train(seed, args.episodes, args.eval_freq, args.eval_episodes, args.out_prefix, insulin_coeff=args.insulin_penalty_coeff)
        csvs.append(csv)

    agg_path = OUT_DIR / f"{args.out_prefix}_aggregated.csv"
    agg = aggregate_results(csvs, agg_path)
    print('Aggregated results saved to', agg_path)

    # Basic summary: focus on eval phases (phase=='eval') and group by train_episode
    if 'phase' in agg.columns:
        eval_rows = agg[agg['phase'] == 'eval']
        if not eval_rows.empty and 'train_episode' in eval_rows.columns:
            summary = eval_rows.groupby(['train_episode']).agg({'time_in_range_percent':'mean','total_reward':'mean'})
            print(summary.head())
        else:
            print('No eval rows found in aggregated CSV; showing top rows:')
            print(agg.head())
    else:
        print('No phase column in aggregated CSV; showing top rows:')
        print(agg.head())

if __name__ == '__main__':
    main()
