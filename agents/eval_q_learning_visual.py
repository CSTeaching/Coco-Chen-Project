"""Visual evaluation of a trained Q-table on the FeverEnv.

Generates and saves:
 - `temp_vs_steps.png`       : Temperature vs step lines per episode
 - `action_counts.png`      : Bar chart counts of actions across all episodes
 - `total_reward_per_episode.png`: Line plot of total reward per episode

Also prints summary statistics (average steps to recovery, counts recovered vs danger).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from env.fever_env import FeverEnv


def discretize_state(temp, edges):
    return int(np.digitize(temp, edges))


def run_visual_eval(q_path="q_table.npz", episodes=20, max_steps=200, seed=0):
    p = Path(q_path)
    if not p.exists():
        raise FileNotFoundError(f"Q-table file not found: {q_path}")

    data = np.load(q_path)
    q_table = data["q_table"]
    edges = data["edges"]

    env = FeverEnv(seed=seed)

    all_temps = []      # list of lists
    all_actions = []    # list of lists
    all_rewards = []    # list of lists
    episode_totals = []
    terminations = []   # 'recovered' or 'danger' or 'max_steps'

    for ep in range(1, episodes + 1):
        temps = []
        actions = []
        rewards = []

        obs = env.reset()
        temp = float(obs[0])
        s = discretize_state(temp, edges)

        temps.append(temp)

        for t in range(max_steps):
            a = int(np.argmax(q_table[s]))
            obs, r, done, info = env.step(a)
            temp = float(obs[0])
            s = discretize_state(temp, edges)

            actions.append(a)
            rewards.append(r)
            temps.append(temp)

            if done:
                terminations.append(info.get('termination', 'done'))
                break
        else:
            terminations.append('max_steps')

        all_temps.append(temps)
        all_actions.append(actions)
        all_rewards.append(rewards)
        episode_totals.append(sum(rewards))

    # Plot 1: Temperature vs Steps for each episode
    plt.figure(figsize=(10, 6))
    for i, temps in enumerate(all_temps, start=1):
        steps = list(range(len(temps)))
        plt.plot(steps, temps, label=f'Ep {i}')
    plt.xlabel('Step')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature vs Steps per Episode')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('temp_vs_steps.png')
    print('Saved temp_vs_steps.png')

    # Plot 2: Action counts across all episodes
    flat_actions = [a for acts in all_actions for a in acts]
    action_labels = ['no med (0)', 'small (1)', 'medium (2)', 'large (3)']
    counts = [flat_actions.count(i) for i in range(4)]

    plt.figure(figsize=(8, 5))
    plt.bar(range(4), counts, tick_label=action_labels)
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title('Action Counts Across Episodes')
    for i, v in enumerate(counts):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig('action_counts.png')
    print('Saved action_counts.png')

    # Plot 3: Total reward per episode
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, episodes + 1), episode_totals, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('total_reward_per_episode.png')
    print('Saved total_reward_per_episode.png')

    # Summary statistics
    recovered_idxs = [i for i, term in enumerate(terminations) if term == 'recovered']
    danger_idxs = [i for i, term in enumerate(terminations) if term == 'danger']

    steps_to_recovery = []
    for idx in recovered_idxs:
        # steps = number of actions taken until termination
        steps_to_recovery.append(len(all_actions[idx]))

    avg_steps = float(np.mean(steps_to_recovery)) if steps_to_recovery else float('nan')

    print('\nSummary:')
    print(f'  Episodes run: {episodes}')
    print(f'  Recovered: {len(recovered_idxs)}')
    print(f'  Danger: {len(danger_idxs)}')
    if not np.isnan(avg_steps):
        print(f'  Average steps to recovery (over recovered episodes): {avg_steps:.2f}')
    else:
        print('  Average steps to recovery: N/A (no recovered episodes)')


if __name__ == '__main__':
    run_visual_eval()
