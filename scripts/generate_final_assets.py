#!/usr/bin/env python3
"""Generate final plots and summary metrics into `final/` from `artifacts/` CSVs.

Produces:
- final/learning_curves.png
- final/eval_boxplot.png
- final/summary_metrics.csv
- final/ (README)
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def read_seed_files(pattern):
    files = sorted(glob.glob(pattern))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['_source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception:
            continue
    return dfs


def compute_eval_summary(dfs):
    # For each seed, compute mean of eval-phase metrics
    rows = []
    for df in dfs:
        name = df['_source_file'].iloc[0]
        eval_df = df[df['phase'] == 'eval']
        if eval_df.empty:
            continue
        mean_glucose = eval_df['mean_glucose'].mean()
        tir = eval_df['time_in_range_percent'].mean()
        reward = eval_df['total_reward'].mean()
        rows.append({'source': name, 'mean_glucose': mean_glucose, 'tir': tir, 'reward': reward})
    return pd.DataFrame(rows)


def make_learning_curve(dfs, out_path):
    # build a wide table of total_reward by train_episode for each seed
    series = []
    for df in dfs:
        s = df[df['phase'] == 'train'][['train_episode','total_reward']].dropna()
        if s.empty:
            continue
        s = s.set_index('train_episode')['total_reward']
        s.name = os.path.basename(df['_source_file'].iloc[0])
        series.append(s)
    if not series:
        return
    df_wide = pd.concat(series, axis=1)
    mean = df_wide.mean(axis=1)
    std = df_wide.std(axis=1)

    plt.figure(figsize=(9,5))
    plt.plot(mean.index, mean.values, label='mean total_reward')
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.3)
    plt.xlabel('train_episode')
    plt.ylabel('total_reward')
    plt.title('Learning curve (mean ± std across seeds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_eval_boxplot(eval_summary_df, out_path, metric='tir'):
    if eval_summary_df.empty:
        return
    plt.figure(figsize=(6,4))
    sns.boxplot(y=eval_summary_df[metric].dropna())
    plt.ylabel(metric)
    plt.title(f'Per-seed distribution of {metric}')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def aggregate_and_save(eval_summary_df, comparison_csv, out_csv):
    # Read existing comparison table if present and append our computed aggregates
    rows = []
    if os.path.exists(comparison_csv):
        try:
            comp = pd.read_csv(comparison_csv)
        except Exception:
            comp = pd.DataFrame()
    else:
        comp = pd.DataFrame()

    if not eval_summary_df.empty:
        tir_mean = eval_summary_df['tir'].mean()
        tir_std = eval_summary_df['tir'].std()
        reward_mean = eval_summary_df['reward'].mean()
        reward_std = eval_summary_df['reward'].std()
        insulin_mean = np.nan
        insulin_std = np.nan
        hypo_mean = np.nan
        severe_mean = np.nan
        n_seeds = len(eval_summary_df)
        our_row = {
            'method': 'Final DQN',
            'tir_mean': tir_mean,
            'tir_std': tir_std,
            'insulin_mean': insulin_mean,
            'insulin_std': insulin_std,
            'reward_mean': reward_mean,
            'reward_std': reward_std,
            'hypo_mean': hypo_mean,
            'severe_mean': severe_mean,
            'n_seeds': n_seeds,
        }
        rows.append(our_row)

    combined = pd.concat([comp, pd.DataFrame(rows)], ignore_index=True, sort=False) if not comp.empty else pd.DataFrame(rows)
    combined.to_csv(out_csv, index=False)
    return combined


def main():
    out_dir = os.path.join(os.getcwd(), 'final')
    ensure_dir(out_dir)

    # Find final DQN seed CSVs
    seed_pattern = os.path.join('artifacts', 'final_dqn_seed*.csv')
    dfs = read_seed_files(seed_pattern)

    eval_summary = compute_eval_summary(dfs)

    # Learning curve
    make_learning_curve(dfs, os.path.join(out_dir, 'learning_curves.png'))

    # Eval boxplot (time-in-range)
    make_eval_boxplot(eval_summary, os.path.join(out_dir, 'eval_tir_boxplot.png'), metric='tir')

    # Eval boxplot (mean_glucose)
    make_eval_boxplot(eval_summary, os.path.join(out_dir, 'eval_mean_glucose_boxplot.png'), metric='mean_glucose')

    # Aggregate comparison
    comparison_csv = os.path.join('artifacts', 'final_comparison_summary.csv')
    out_csv = os.path.join(out_dir, 'summary_metrics.csv')
    combined = aggregate_and_save(eval_summary, comparison_csv, out_csv)

    # Save per-seed summary for transparency
    eval_summary.to_csv(os.path.join(out_dir, 'per_seed_eval_summary.csv'), index=False)

    print('Wrote final assets to', out_dir)


if __name__ == '__main__':
    main()
