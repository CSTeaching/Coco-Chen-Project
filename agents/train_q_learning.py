"""Train a Q-learning agent on the FeverEnv.

Saves Q-table and bin edges to `q_table.npz`.

Usage:
    python train_q_learning.py --episodes 2000 --bins 20
"""

import argparse
import numpy as np
from collections import deque

from env.fever_env import FeverEnv


def discretize_state(temp, edges):
    """Map scalar temperature to discrete bin index."""
    # np.digitize returns 1..len(edges) indices for bins between edges
    idx = np.digitize(temp, edges)
    # ensure index in [0, n_bins-1]
    return int(idx)


def train(episodes=2000, bins=20, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.9995, max_steps=200, seed=0, mode='probabilistic'):
    env = FeverEnv(seed=seed, mode=mode)

    # build bin edges so we have `bins` discrete states between 36 and 41
    edges = np.linspace(36.0, 41.0, bins + 1)[1:-1]
    n_states = bins
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    eps = eps_start
    rewards_window = deque(maxlen=100)

    for ep in range(1, episodes + 1):
        obs = env.reset()
        temp = float(obs[0])
        s = discretize_state(temp, edges)
        total_reward = 0.0

        for t in range(max_steps):
            # epsilon-greedy
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(q_table[s]))

            obs2, r, done, info = env.step(a)
            temp2 = float(obs2[0])
            s2 = discretize_state(temp2, edges)

            # Q-learning update
            best_next = np.max(q_table[s2])
            q_table[s, a] += alpha * (r + gamma * best_next - q_table[s, a])

            s = s2
            total_reward += r
            if done:
                break

        rewards_window.append(total_reward)

        # decay epsilon
        eps = max(eps_end, eps * eps_decay)

        if ep % 100 == 0:
            avg_reward = float(np.mean(rewards_window)) if rewards_window else 0.0
            print(f"Episode {ep:5d}/{episodes}  avg_reward(last{len(rewards_window)}): {avg_reward:.3f}  eps: {eps:.3f}")

    # save q-table and edges
    np.savez("q_table.npz", q_table=q_table, edges=edges)
    print("Saved Q-table to q_table.npz")
    return q_table, edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default='probabilistic', choices=['baseline','probabilistic','shaped_reward','strong_penalty'])
    args = parser.parse_args()

    train(episodes=args.episodes, bins=args.bins, seed=args.seed, mode=args.mode)


if __name__ == "__main__":
    main()
