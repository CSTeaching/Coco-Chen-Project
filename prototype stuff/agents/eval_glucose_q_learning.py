"""Evaluate a trained Q-learning glucose agent with greedy policy.

Loads a saved Q-table and runs episodes, printing glucose traces and metrics.

Usage:
    python agents/eval_glucose_q_learning.py --q_table glucose_q_table_standard_type2_stable.npz --episodes 5
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.glucose_env import GlucoseEnv


def discretize_state(glucose, edges):
    return int(np.clip(np.digitize(glucose, edges), 0, len(edges) - 1))


def evaluate(q_table_path, episodes=5, max_steps=288, seed=0):
    """Run greedy evaluation.
    
    Args:
        q_table_path: path to saved Q-table .npz file
        episodes: number of evaluation episodes
        max_steps: max steps per episode
        seed: random seed
    """
    # Load Q-table
    data = np.load(q_table_path)
    q_table = data["q_table"]
    edges = data["edges"]
    mode = str(data.get("mode", "standard"))
    patient = str(data.get("patient", "type2_stable"))

    env = GlucoseEnv(seed=seed, mode=mode, patient_type=patient)

    print(f"\nEvaluating Q-table from: {q_table_path}")
    print(f"  Mode: {mode}, Patient: {patient}")
    print(f"  Episodes: {episodes}\n")

    episode_metrics = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        glucose = float(obs[0])
        s = discretize_state(glucose, edges)

        print(f"Episode {ep} | Start glucose: {glucose:6.1f} mg/dL")

        glucose_trace = [glucose]
        action_trace = []
        reward_trace = []
        time_in_range = 0

        for t in range(max_steps):
            # Greedy action
            a = int(np.argmax(q_table[s]))
            obs2, r, done, info = env.step(a)
            glucose = float(obs2[0])
            s = discretize_state(glucose, edges)

            glucose_trace.append(glucose)
            action_trace.append(a)
            reward_trace.append(r)

            if 80 <= glucose <= 180:
                time_in_range += 1

            if done:
                print(f"  Step {t:3d}: insulin={a*2:2d}U, glucose={glucose:6.1f}, reward={r:7.2f} -> {info.get('termination', 'done')}")
                break
            elif t < 5 or t % 20 == 0:
                print(f"  Step {t:3d}: insulin={a*2:2d}U, glucose={glucose:6.1f}, reward={r:7.2f}")

        # Metrics for this episode
        avg_glucose = np.mean(glucose_trace)
        min_glucose = np.min(glucose_trace)
        max_glucose = np.max(glucose_trace)
        time_in_range_pct = 100 * time_in_range / len(glucose_trace) if glucose_trace else 0
        total_reward = np.sum(reward_trace)
        episode_len = len(action_trace)

        episode_metrics.append({
            'episode': ep,
            'avg_glucose': avg_glucose,
            'min_glucose': min_glucose,
            'max_glucose': max_glucose,
            'time_in_range_pct': time_in_range_pct,
            'total_reward': total_reward,
            'episode_length': episode_len
        })

        print(f"  Summary: avg={avg_glucose:.1f}, min={min_glucose:.1f}, max={max_glucose:.1f}, "
              f"time_in_range={time_in_range_pct:.1f}%, total_reward={total_reward:.1f}\n")

    # Print aggregate statistics
    print("=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)
    avg_glucose_all = np.mean([m['avg_glucose'] for m in episode_metrics])
    avg_min_glucose = np.mean([m['min_glucose'] for m in episode_metrics])
    avg_max_glucose = np.mean([m['max_glucose'] for m in episode_metrics])
    avg_time_in_range = np.mean([m['time_in_range_pct'] for m in episode_metrics])
    avg_reward_all = np.mean([m['total_reward'] for m in episode_metrics])

    print(f"Average glucose (over episodes): {avg_glucose_all:.1f} mg/dL")
    print(f"Average min glucose: {avg_min_glucose:.1f} mg/dL")
    print(f"Average max glucose: {avg_max_glucose:.1f} mg/dL")
    print(f"Average time in range (80-180): {avg_time_in_range:.1f}%")
    print(f"Average total reward: {avg_reward_all:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained glucose Q-learning agent")
    parser.add_argument("--q_table", type=str, default="glucose_q_table_standard_type2_stable.npz")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evaluate(args.q_table, episodes=args.episodes, seed=args.seed)
