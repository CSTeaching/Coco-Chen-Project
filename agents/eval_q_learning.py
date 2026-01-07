"""Evaluate a trained Q-table on the FeverEnv with a greedy policy.

Loads `q_table.npz` produced by `train_q_learning.py` and runs 20 episodes,
printing actions and temperatures at each step.
"""

import numpy as np
from env.fever_env import FeverEnv


def discretize_state(temp, edges):
    return int(np.digitize(temp, edges))


def evaluate(q_path="q_table.npz", episodes=20, max_steps=200, seed=0):
    data = np.load(q_path)
    q_table = data["q_table"]
    edges = data["edges"]

    env = FeverEnv(seed=seed)

    for ep in range(1, episodes + 1):
        obs = env.reset()
        temp = float(obs[0])
        s = discretize_state(temp, edges)
        print(f"Episode {ep} start temp: {temp:.2f}°C")

        for t in range(max_steps):
            a = int(np.argmax(q_table[s]))
            obs, r, done, info = env.step(a)
            temp = float(obs[0])
            s = discretize_state(temp, edges)
            print(f"  step {t:03d}: action={a}, temp={temp:.2f}°C, reward={r:.2f}")
            if done:
                print(f"  -> done ({info.get('termination')})\n")
                break
        else:
            print("  -> reached max steps\n")


if __name__ == "__main__":
    evaluate()
