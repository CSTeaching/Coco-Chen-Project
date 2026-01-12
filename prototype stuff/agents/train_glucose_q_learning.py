"""Train a Q-learning agent on the glucose management environment.

Saves Q-table and bin edges to `glucose_q_table.npz`.

Usage:
    python agents/train_glucose_q_learning.py --episodes 2000 --bins 30 --mode standard --patient type2_stable
"""

import argparse
import numpy as np
from collections import deque
import sys
import os

# Add parent directory to path to import env module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.glucose_env import GlucoseEnv


def discretize_state(glucose, edges):
    """Map glucose level to discrete bin index."""
    idx = np.digitize(glucose, edges)
    return int(np.clip(idx, 0, len(edges) - 1))


def train(episodes=2000, bins=30, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, 
          eps_decay=0.9995, max_steps=288, seed=0, mode='standard', patient='type2_stable'):
    """Train Q-learning agent on glucose environment.
    
    Args:
        episodes: number of training episodes
        bins: number of glucose level bins for discretization
        alpha: learning rate
        gamma: discount factor
        eps_start, eps_end, eps_decay: epsilon-greedy parameters
        max_steps: max steps per episode (288 = 24 hours at 5-min intervals)
        seed: random seed
        mode: reward mode ('standard', 'aggressive_reward', 'conservative_reward', 'variable_meals')
        patient: patient type ('type2_stable', 'type2_variable', 'brittle_diabetes')
    """
    env = GlucoseEnv(seed=seed, mode=mode, patient_type=patient)

    # Build bin edges: glucose range 40-400 mg/dL
    edges = np.linspace(40.0, 400.0, bins + 1)[1:-1]
    n_states = bins
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    eps = eps_start
    rewards_window = deque(maxlen=100)

    print(f"Training glucose Q-learning agent")
    print(f"  Mode: {mode}")
    print(f"  Patient: {patient}")
    print(f"  Episodes: {episodes}")
    print(f"  Bins: {bins}")
    print(f"  Alpha: {alpha}, Gamma: {gamma}\n")

    for ep in range(1, episodes + 1):
        obs = env.reset()
        glucose = float(obs[0])
        s = discretize_state(glucose, edges)
        total_reward = 0.0

        for t in range(max_steps):
            # Epsilon-greedy action selection
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(q_table[s]))

            obs2, r, done, info = env.step(a)
            glucose2 = float(obs2[0])
            s2 = discretize_state(glucose2, edges)

            # Q-learning update
            best_next = np.max(q_table[s2])
            q_table[s, a] += alpha * (r + gamma * best_next - q_table[s, a])

            s = s2
            total_reward += r
            if done:
                break

        rewards_window.append(total_reward)

        # Decay epsilon
        eps = max(eps_end, eps * eps_decay)

        if ep % 100 == 0:
            avg_reward = float(np.mean(rewards_window)) if rewards_window else 0.0
            print(f"Episode {ep:5d}/{episodes}  avg_reward(last100): {avg_reward:8.2f}  eps: {eps:.3f}")

    # Save Q-table
    filename = f"glucose_q_table_{mode}_{patient}.npz"
    np.savez(filename, q_table=q_table, edges=edges, mode=mode, patient=patient)
    print(f"\nSaved Q-table to {filename}")
    return q_table, edges


def main():
    parser = argparse.ArgumentParser(description="Train Q-learning agent on glucose environment")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default='standard',
                        choices=['standard', 'aggressive_reward', 'conservative_reward', 'variable_meals'])
    parser.add_argument("--patient", type=str, default='type2_stable',
                        choices=['type2_stable', 'type2_variable', 'brittle_diabetes'])
    args = parser.parse_args()

    train(episodes=args.episodes, bins=args.bins, seed=args.seed, mode=args.mode, patient=args.patient)


if __name__ == "__main__":
    main()
