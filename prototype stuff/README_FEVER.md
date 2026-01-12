# Fever Gym Environment (toy)

This toy environment simulates fever dynamics for reinforcement learning experiments. It has been updated to include stochastic medicine effects, side-effects, and variable episode starts so agents must learn safer, multi-step dosing strategies.

**Key Behavior (updated)**
- **Temperature range:** 36.0°C — 41.0°C (observation is a single scalar `np.array([temp])`).
- **Actions (`Discrete(4)`):**
  - `0` = no medicine → natural small increase (+0.1°C).
  - `1` = small dose  → decreases temperature by **uniform(0.8, 1.2)** °C.
  - `2` = medium dose → decreases temperature by **uniform(1.3, 1.7)** °C.
  - `3` = large dose  → decreases temperature by **uniform(1.8, 2.2)** °C.

- **Side-effects:** each dose has a chance to *increase* temperature by **uniform(0.5, 1.0)** °C:
  - small dose: 5% chance
  - medium dose: 10% chance
  - large dose: 20% chance
  - When a side-effect occurs, it is recorded in `info['side_effect']` and incurs a **-1 reward** penalty.

- **Variable starting temperature:** `reset()` samples the episode start temperature uniformly from **[38.0, 40.0] °C** (previously fixed at 39.0°C).

- **Cap on per-step drop:** no single step may reduce temperature below **36.0°C**; if a dose would, the delta is adjusted so temperature reaches exactly 36.0°C (prevents instant extreme hypothermia in one step).

- **Termination:**
  - Danger: `temp < 36.0` or `temp > 41.0` → episode ends (heavy penalty).
  - Recovered: `36.9 <= temp <= 37.1` → episode ends (bonus).

- **Rewards:**
  - Base shaping: `-abs(temp - 37.0)` (encourages closeness to 37°C).
  - Recovery bonus: **+10** when recovered.
  - Danger penalty: **-10** when danger occurs.
  - Additional **-5** penalty if `temp < 36.0` (severe hypothermia).
  - Side-effect penalty: **-1**.

All stochasticity uses the environment's RNG (`seed` parameter in `FeverEnv`) so runs can be made reproducible.

**Files & Scripts**
- `fever_env.py`: the updated `FeverEnv` class (Gym API: `reset()` and `step(action)` returning `(obs, reward, done, info)`).
- `train_q_learning.py`: Q-learning trainer that discretizes temperature into bins, trains a Q-table, prints average reward every 100 episodes, and saves `q_table.npz`.
- `eval_q_learning.py`: loads `q_table.npz` and runs greedy evaluation episodes, printing actions and temperatures.
- `eval_q_learning_visual.py`: runs greedy episodes, records temperatures/actions/rewards, and saves three plots:
  - `temp_vs_steps.png` (temperature vs step for each episode)
  - `action_counts.png` (bar chart of action counts)
  - `total_reward_per_episode.png` (total reward per episode)
- `example_random_agent.py`: simple random agent demo.
- `requirements.txt`: minimal dependencies (`gym`, `numpy`, plus `matplotlib` if you want plotting).

**Quick start**
1. Create and activate a virtualenv, install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# install matplotlib for visualization
pip install matplotlib
```

2. Train a Q-table (example):

```bash
python train_q_learning.py --episodes 2000 --bins 20
```

3. Evaluate greedy policy and print episode traces:

```bash
python eval_q_learning.py
```

4. Produce visual evaluation (saves PNGs):

```bash
python eval_q_learning_visual.py
```

Outputs created by training/evaluation
- `q_table.npz`: saved Q-table and bin edges (created by `train_q_learning.py`).
- `temp_vs_steps.png`, `action_counts.png`, `total_reward_per_episode.png`: created by `eval_q_learning_visual.py`.

**Notes & suggestions**
- The environment was purposefully made more stochastic to force multi-step, safer dosing policies. Agents will typically require more training and possibly reward/hyperparameter tuning to reach good performance.
- `gym` is unmaintained and prints a migration suggestion; you may switch to `gymnasium` if you want a maintained API. In most cases, replacing `import gym` with `import gymnasium as gym` works, but check compatibility.
- If you want, I can:
  - Register the environment (so `gym.make("Fever-v0")` works),
  - Add an evaluation CSV logger or save per-episode traces to a JSON/CSV, or
  - Add a short README section with recommended hyperparameters.
