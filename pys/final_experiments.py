#!/usr/bin/env python3
"""Final multi-seed DQN training and baseline comparison for paper.

Runs:
1. Multi-seed DQN training with best-checkpoint selection
2. Baseline evaluation on identical seeds
3. Aggregate results across seeds
4. Generate comparison tables and plots

Output:
- artifacts/final_dqn_results.csv (per-episode metrics)
- artifacts/final_dqn_summary.csv (aggregate across seeds)
- artifacts/final_baseline_results.csv (baseline metrics)
- artifacts/final_comparison_summary.csv (DQN vs baselines summary)
- plots/final/ (comparison visualizations)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pys.glucose_env import GlucoseEnv
from pys.simulator_params import load_params
from baselines.eval_baselines import (
    NoInsulinPolicy,
    SingleThresholdPolicy,
    TwoThresholdPolicy,
    evaluate_baseline,
)


def run_all_final_experiments(n_seeds: int = 5, dqn_episodes: int = 1000, baseline_episodes: int = 100) -> None:
    """Run all final experiments and generate summary tables.
    
    Args:
        n_seeds: Number of random seeds to use (default 5 for paper).
        dqn_episodes: Episodes per DQN seed (default 1000).
        baseline_episodes: Episodes per baseline seed (default 100).
    """
    seeds = list(range(42, 42 + n_seeds))
    params = load_params()
    out_dir = Path("artifacts")
    plots_dir = Path("plots/final")
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("FINAL MULTI-SEED DQN vs BASELINE EXPERIMENTS")
    print("=" * 90)
    print(f"Seeds: {seeds}")
    print(f"DQN episodes per seed: {dqn_episodes}")
    print(f"Baseline episodes per seed: {baseline_episodes}")
    print("=" * 90 + "\n")

    # ========== Run DQN for each seed ==========
    print("[1/3] Running multi-seed DQN training...")
    dqn_results = []
    for seed_idx, seed in enumerate(seeds):
        print(f"  DQN seed {seed_idx+1}/{n_seeds} (seed={seed})")
        
        # Train DQN with early stopping
        import subprocess
        cmd = [
            sys.executable,
            "agents/train_dqn.py",
            "--episodes", str(dqn_episodes),
            "--eval-freq", "50",
            "--eval-episodes", "5",
            "--insulin-penalty-coeff", "0.1",
            "--early-stop-patience", "5",
            "--early-stop-min-evals", "4",
            "--safety-hypo-max", "3.0",
            "--safety-severe-max", "2.0",
            "--seed", str(seed),
            "--out", f"artifacts/dqn_seed_{seed}.csv"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspaces/Project")
        if result.returncode != 0:
            print(f"    Error running DQN seed {seed}: {result.stderr[:200]}")
            continue
        
        # Parse output CSV
        csv_path = Path(f"artifacts/dqn_seed_{seed}.csv")
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["seed"] = seed
                    dqn_results.append(row)
            print(f"    ✓ Completed, best checkpoint saved")
    
    # Save DQN raw results
    if dqn_results:
        with open(out_dir / "final_dqn_results.csv", "w", newline="") as f:
            fieldnames = list(dqn_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dqn_results)
        print(f"  ✓ Saved {len(dqn_results)} DQN episodes to {out_dir / 'final_dqn_results.csv'}\n")

    # ========== Run baselines for same seeds ==========
    print("[2/3] Running baseline evaluation on same seeds...")
    baseline_results = []
    env = GlucoseEnv(random_seed=seeds[0])
    policies = [
        NoInsulinPolicy(env),
        SingleThresholdPolicy(env, params),
        TwoThresholdPolicy(env, params),
    ]
    
    for policy in policies:
        print(f"  Evaluating {policy.name}...")
        for seed_idx, seed in enumerate(seeds):
            episodes = evaluate_baseline(policy, env, baseline_episodes, seed, params)
            baseline_results.extend(episodes)
        print(f"    ✓ Completed {len(seeds)} seeds")
    
    # Save baseline results
    if baseline_results:
        with open(out_dir / "final_baseline_results.csv", "w", newline="") as f:
            fieldnames = list(baseline_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(baseline_results)
        print(f"  ✓ Saved {len(baseline_results)} baseline episodes to {out_dir / 'final_baseline_results.csv'}\n")

    # ========== Aggregate and summarize ==========
    print("[3/3] Generating summary tables...")
    
    # DQN summary (best checkpoint only, per seed)
    dqn_by_seed = {}
    for row in dqn_results:
        seed = row["seed"]
        if seed not in dqn_by_seed:
            dqn_by_seed[seed] = []
        dqn_by_seed[seed].append(row)
    
    dqn_summary = []
    for seed in seeds:
        if seed in dqn_by_seed:
            rows = dqn_by_seed[seed]
            # Take the best row (lowest episode, which is the best checkpoint saved)
            best_row = min(rows, key=lambda r: int(r.get("episode", 999)))
            dqn_summary.append({
                "seed": seed,
                "mean_glucose": float(best_row.get("mean_glucose", 0)),
                "tir": float(best_row.get("time_in_range_percent", 0)),
                "hypo": float(best_row.get("hypo_count", 0)),
                "severe": float(best_row.get("severe_hyper_count", 0)),
                "insulin": float(best_row.get("total_insulin_units", 0)),
                "reward": float(best_row.get("total_reward", 0)),
            })
    
    if dqn_summary:
        with open(out_dir / "final_dqn_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dqn_summary[0].keys()))
            writer.writeheader()
            writer.writerows(dqn_summary)
    
    # Baseline + DQN comparison
    comparison = []
    
    # DQN aggregate
    if dqn_summary:
        dqn_tir = np.mean([float(r["tir"]) for r in dqn_summary])
        dqn_tir_std = np.std([float(r["tir"]) for r in dqn_summary])
        dqn_insulin = np.mean([float(r["insulin"]) for r in dqn_summary])
        dqn_insulin_std = np.std([float(r["insulin"]) for r in dqn_summary])
        dqn_reward = np.mean([float(r["reward"]) for r in dqn_summary])
        dqn_reward_std = np.std([float(r["reward"]) for r in dqn_summary])
        dqn_hypo = np.mean([float(r["hypo"]) for r in dqn_summary])
        dqn_severe = np.mean([float(r["severe"]) for r in dqn_summary])
        
        comparison.append({
            "method": "DQN (frozen simulator, coeff=0.1)",
            "tir_mean": f"{dqn_tir:.1f}",
            "tir_std": f"{dqn_tir_std:.1f}",
            "insulin_mean": f"{dqn_insulin:.1f}",
            "insulin_std": f"{dqn_insulin_std:.1f}",
            "reward_mean": f"{dqn_reward:.1f}",
            "reward_std": f"{dqn_reward_std:.1f}",
            "hypo_mean": f"{dqn_hypo:.1f}",
            "severe_mean": f"{dqn_severe:.1f}",
            "n_seeds": len(dqn_summary),
        })
    
    # Baseline aggregates
    baseline_by_policy = {}
    for row in baseline_results:
        policy = row["baseline"]
        if policy not in baseline_by_policy:
            baseline_by_policy[policy] = []
        baseline_by_policy[policy].append(row)
    
    for policy in sorted(baseline_by_policy.keys()):
        rows = baseline_by_policy[policy]
        tir_vals = [float(r["time_in_range_percent"]) for r in rows]
        insulin_vals = [float(r["total_insulin_units"]) for r in rows]
        reward_vals = [float(r["total_reward"]) for r in rows]
        hypo_vals = [float(r["hypo_count"]) for r in rows]
        severe_vals = [float(r["severe_hyper_count"]) for r in rows]
        
        comparison.append({
            "method": f"Baseline: {policy}",
            "tir_mean": f"{np.mean(tir_vals):.1f}",
            "tir_std": f"{np.std(tir_vals):.1f}",
            "insulin_mean": f"{np.mean(insulin_vals):.1f}",
            "insulin_std": f"{np.std(insulin_vals):.1f}",
            "reward_mean": f"{np.mean(reward_vals):.1f}",
            "reward_std": f"{np.std(reward_vals):.1f}",
            "hypo_mean": f"{np.mean(hypo_vals):.1f}",
            "severe_mean": f"{np.mean(severe_vals):.1f}",
            "n_seeds": len([r for r in rows if r["seed"] in seeds]),
        })
    
    if comparison:
        with open(out_dir / "final_comparison_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
            writer.writeheader()
            writer.writerows(comparison)
    
    # Print summary table
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Method':<40} {'TIR %':<15} {'Insulin U':<15} {'Reward':<15} {'N Seeds':<10}")
    print("-" * 100)
    for row in comparison:
        print(f"{row['method']:<40} {row['tir_mean']:>6} ± {row['tir_std']:<6} "
              f"{row['insulin_mean']:>6} ± {row['insulin_std']:<6} "
              f"{row['reward_mean']:>6} ± {row['reward_std']:<6} {row['n_seeds']:>10}")
    print("=" * 100)
    
    print(f"\n✓ Saved all results to {out_dir}/")
    print(f"✓ Summary table saved to {out_dir / 'final_comparison_summary.csv'}")
    print("\nNext: Generate visualization plots with plots_final.py")


if __name__ == "__main__":
    n_seeds = 5  # Can increase to 10 for more robust statistics
    dqn_episodes = 1000
    baseline_episodes = 100
    
    run_all_final_experiments(n_seeds=n_seeds, dqn_episodes=dqn_episodes, baseline_episodes=baseline_episodes)
