"""Visual evaluation of glucose management Q-learning agent.

Generates plots showing glucose traces, insulin actions, and reward patterns
across multiple evaluation episodes.

Usage:
    python agents/eval_glucose_visual.py --q_table glucose_q_table_standard_type2_stable.npz --episodes 10
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.glucose_env import GlucoseEnv


def discretize_state(glucose, edges):
    return int(np.clip(np.digitize(glucose, edges), 0, len(edges) - 1))


def run_visual_eval(q_table_path, episodes=10, max_steps=288, seed=0):
    """Run evaluation and generate visualizations."""
    data = np.load(q_table_path)
    q_table = data["q_table"]
    edges = data["edges"]
    mode = str(data.get("mode", "standard"))
    patient = str(data.get("patient", "type2_stable"))

    env = GlucoseEnv(seed=seed, mode=mode, patient_type=patient)

    print(f"\nVisual evaluation: {q_table_path}")
    print(f"  Mode: {mode}, Patient: {patient}")
    print(f"  Episodes: {episodes}\n")

    all_glucose_traces = []
    all_insulin_traces = []
    all_reward_traces = []
    episode_metrics = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        glucose = float(obs[0])
        s = discretize_state(glucose, edges)

        glucose_trace = [glucose]
        insulin_trace = []
        reward_trace = []
        time_in_range = 0

        for t in range(max_steps):
            a = int(np.argmax(q_table[s]))
            insulin_dose = a * 2
            obs2, r, done, info = env.step(a)
            glucose = float(obs2[0])
            s = discretize_state(glucose, edges)

            glucose_trace.append(glucose)
            insulin_trace.append(insulin_dose)
            reward_trace.append(r)

            if 80 <= glucose <= 180:
                time_in_range += 1

            if done:
                break

        all_glucose_traces.append(glucose_trace)
        all_insulin_traces.append(insulin_trace)
        all_reward_traces.append(reward_trace)

        avg_glucose = np.mean(glucose_trace)
        min_glucose = np.min(glucose_trace)
        max_glucose = np.max(glucose_trace)
        time_in_range_pct = 100 * time_in_range / len(glucose_trace)
        total_reward = np.sum(reward_trace)

        episode_metrics.append({
            'avg': avg_glucose,
            'min': min_glucose,
            'max': max_glucose,
            'tir': time_in_range_pct,
            'reward': total_reward,
            'steps': len(insulin_trace)
        })

        print(f"Episode {ep:2d}: avg={avg_glucose:6.1f}, min={min_glucose:6.1f}, max={max_glucose:6.1f}, "
              f"TIR={time_in_range_pct:5.1f}%, reward={total_reward:7.1f}")

    # --- Plot 1: Glucose traces over time ---
    plt.figure(figsize=(14, 6))
    for i, trace in enumerate(all_glucose_traces[:5], 1):  # Plot first 5 episodes
        time_steps = np.arange(len(trace))
        plt.plot(time_steps, trace, label=f'Episode {i}', alpha=0.7)
    plt.axhline(y=80, color='green', linestyle='--', label='Safe min (80)', alpha=0.5)
    plt.axhline(y=180, color='red', linestyle='--', label='Safe max (180)', alpha=0.5)
    plt.xlabel('Time Step (5-min intervals)')
    plt.ylabel('Glucose (mg/dL)')
    plt.title(f'Glucose Traces - {mode.replace("_", " ").title()} ({patient.replace("_", " ").title()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('glucose_traces.png', dpi=100)
    print("\nSaved glucose_traces.png")

    # --- Plot 2: Insulin doses over time ---
    plt.figure(figsize=(14, 6))
    for i, trace in enumerate(all_insulin_traces[:5], 1):
        time_steps = np.arange(len(trace))
        plt.step(time_steps, trace, label=f'Episode {i}', alpha=0.7, where='post')
    plt.xlabel('Time Step (5-min intervals)')
    plt.ylabel('Insulin Dose (units)')
    plt.title(f'Insulin Actions - {mode.replace("_", " ").title()} ({patient.replace("_", " ").title()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('insulin_actions.png', dpi=100)
    print("Saved insulin_actions.png")

    # --- Plot 3: Reward traces ---
    plt.figure(figsize=(14, 6))
    for i, trace in enumerate(all_reward_traces[:5], 1):
        time_steps = np.arange(len(trace))
        plt.plot(time_steps, trace, label=f'Episode {i}', alpha=0.7)
    plt.xlabel('Time Step (5-min intervals)')
    plt.ylabel('Reward')
    plt.title(f'Reward Traces - {mode.replace("_", " ").title()} ({patient.replace("_", " ").title()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_traces.png', dpi=100)
    print("Saved reward_traces.png")

    # --- Plot 4: Episode metrics (box plots) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Average glucose
    avg_glucoses = [m['avg'] for m in episode_metrics]
    axes[0, 0].hist(avg_glucoses, bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(x=120, color='green', linestyle='--', label='Target (120)')
    axes[0, 0].set_xlabel('Average Glucose (mg/dL)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Average Glucose Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Min/Max glucose
    min_glucoses = [m['min'] for m in episode_metrics]
    max_glucoses = [m['max'] for m in episode_metrics]
    axes[0, 1].scatter(min_glucoses, max_glucoses, alpha=0.6, s=100)
    axes[0, 1].axhline(y=180, color='red', linestyle='--', label='Safe max')
    axes[0, 1].axvline(x=80, color='green', linestyle='--', label='Safe min')
    axes[0, 1].set_xlabel('Min Glucose (mg/dL)')
    axes[0, 1].set_ylabel('Max Glucose (mg/dL)')
    axes[0, 1].set_title('Min vs Max Glucose per Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Time in range
    tir_values = [m['tir'] for m in episode_metrics]
    axes[1, 0].bar(range(len(tir_values)), tir_values, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axhline(y=70, color='orange', linestyle='--', label='Target ≥70%')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Time in Range (%)')
    axes[1, 0].set_title('Time in Range (80-180 mg/dL) per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 105])

    # Total reward
    rewards = [m['reward'] for m in episode_metrics]
    axes[1, 1].bar(range(len(rewards)), rewards, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].set_title('Total Reward per Episode')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('glucose_metrics.png', dpi=100)
    print("Saved glucose_metrics.png")

    # --- Summary statistics ---
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Episodes evaluated: {episodes}")
    print(f"  Average glucose (mean ± std): {np.mean(avg_glucoses):.1f} ± {np.std(avg_glucoses):.1f} mg/dL")
    print(f"  Average minimum glucose: {np.mean(min_glucoses):.1f} mg/dL")
    print(f"  Average maximum glucose: {np.mean(max_glucoses):.1f} mg/dL")
    print(f"  Average time in range (80-180): {np.mean(tir_values):.1f}%")
    print(f"  Average total reward: {np.mean(rewards):.1f}")
    print(f"  Average episode length: {np.mean([m['steps'] for m in episode_metrics]):.0f} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual evaluation of glucose Q-learning agent")
    parser.add_argument("--q_table", type=str, default="glucose_q_table_standard_type2_stable.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_visual_eval(args.q_table, episodes=args.episodes, seed=args.seed)
