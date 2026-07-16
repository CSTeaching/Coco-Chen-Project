"""DQN training for glucose control with improved metrics tracking.

Trains a Deep Q-Network to learn insulin dosing policies.
Now with accurate metrics: separate step/event counting, insulin accounting (bolus vs basal),
action histograms, and reward breakdowns.

Metrics:
  - hypo_steps: count of timesteps < 66 mg/dL
  - hypo_events: count of transitions into hypo zone
  - severe_hyper_steps: count of timesteps > 292 mg/dL
  - severe_hyper_events: count of transitions into severe hyper zone
  - total_bolus_units: sum of bolus doses delivered
  - total_basal_units: basal_rate * 24 hours
  - total_insulin_units: bolus + basal
  - action_counts: histogram of actions [0, 1, 2, 3, 4]
  - sum_zone_reward: reward from glucose zones
  - sum_insulin_penalty: penalty from insulin use
  - total_reward: cumulative reward
  - plus glucose control metrics (mean, TIR)

Usage:
    python agents/train_dqn.py --episodes 1000 --seed 42 --out artifacts/dqn_training.csv
"""

import os
import sys
import argparse
import csv
import numpy as np
from collections import deque
from typing import Tuple, Dict, List

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency guard
    torch = None
    nn = None
    optim = None
    F = None
    TORCH_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pys.glucose_env import GlucoseEnv


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer with random sampling."""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in batch:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DQN Network
# ============================================================================

class DQN(nn.Module):
    """DQN for glucose control: (5-dim state) -> (5 Q-values)."""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_actions: int = 5):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, state):
        return self.net(state)


# ============================================================================
# DQN Agent
# ============================================================================

class DQNAgent:
    """DQN agent with epsilon-greedy exploration."""
    
    def __init__(
        self,
        env: GlucoseEnv,
        history_len: int = 6,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 500,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.env = env
        self.device = torch.device(device)
        self.rng = np.random.RandomState(seed)
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        
        self.history_len = history_len
        obs_dim = int(np.prod(self.env.observation_space.shape))
        input_dim = obs_dim * self.history_len
        self.q_net = DQN(input_dim=input_dim, hidden_dim=128, num_actions=5).to(self.device)
        self.target_net = DQN(input_dim=input_dim, hidden_dim=128, num_actions=5).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        
        self.total_steps = 0
        # action smoothing to avoid large discrete jumps (improves stability)
        self.last_action = None
        self.max_action_jump = 1
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        # Choose an action (epsilon-greedy) then apply optional smoothing
        if self.rng.rand() < epsilon:
            action = int(self.env.action_space.sample())
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(state_tensor)
                action = int(q_values.argmax(dim=1).item())

        # Apply action smoothing: limit the discrete jump size between steps
        if self.last_action is None:
            smoothed = action
        else:
            delta = action - int(self.last_action)
            if abs(delta) > self.max_action_jump:
                smoothed = int(self.last_action) + int(np.sign(delta) * self.max_action_jump)
            else:
                smoothed = action

        self.last_action = smoothed
        return smoothed
    
    def train_step(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def get_epsilon(self, episode: int) -> float:
        progress = min(episode / self.epsilon_decay_episodes, 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress


# ============================================================================
# Metrics Computation (IMPROVED)
# ============================================================================

def compute_episode_metrics(
    glucose_values: List[float],
    actions_taken: List[int],
    hypo_steps: List[int],
    hypo_events: List[int],
    severe_hyper_steps: List[int],
    severe_hyper_events: List[int],
    bolus_values: List[float],
    zone_rewards: List[float],
    insulin_penalties: List[float],
    total_rewards: List[float],
    basal_rate: float,
    bolus_actions: np.ndarray,
    dt_minutes: float,
) -> Dict[str, float]:
    """Compute comprehensive episode metrics with accurate accounting.
    
    Args:
        glucose_values: List of glucose readings per step
        actions_taken: List of action indices per step
        hypo_steps: List of 1/0 indicating hypo zone per step
        hypo_events: List of 1/0 indicating hypo TRANSITION per step
        severe_hyper_steps: List of 1/0 indicating severe hyper zone per step
        severe_hyper_events: List of 1/0 indicating severe hyper TRANSITION per step
        bolus_values: List of bolus doses per step
        zone_rewards: List of zone reward component per step
        insulin_penalties: List of insulin penalty component per step
        total_rewards: List of total reward per step
        basal_rate: Basal rate from environment (U/hr)
        bolus_actions: Array of action values [0, 2.8, 4.8, 7.6, 12.5]
    
    Returns:
        Dict with all metrics.
    """
    glucose_values = np.array(glucose_values)
    bolus_values = np.array(bolus_values)
    actions_taken = np.array(actions_taken)
    
    # Glucose control metrics
    mean_glucose = float(np.mean(glucose_values))
    time_in_range = float(np.mean((glucose_values >= 80) & (glucose_values <= 180)) * 100)
    
    # Hypo metrics (steps vs events)
    hypo_steps_count = int(np.sum(hypo_steps))
    hypo_events_count = int(np.sum(hypo_events))
    
    # Severe hyper metrics (steps vs events)
    severe_hyper_steps_count = int(np.sum(severe_hyper_steps))
    severe_hyper_events_count = int(np.sum(severe_hyper_events))
    
    # Insulin accounting
    # Use actual episode duration (steps * dt_minutes) for unit-correct basal totals.
    episode_hours = (len(total_rewards) * dt_minutes) / 60.0
    total_bolus_units = float(np.sum(bolus_values))
    total_basal_units = float(basal_rate * episode_hours)  # U/hr * hours
    total_insulin_units = total_bolus_units + total_basal_units
    
    # Action histogram
    action_counts = {}
    for i, dose in enumerate(bolus_actions):
        count = int(np.sum(actions_taken == i))
        action_counts[f"action_{i}_count"] = count
    
    # Reward breakdown
    sum_zone_reward = float(np.sum(zone_rewards))
    sum_insulin_penalty = float(np.sum(insulin_penalties))
    total_reward = float(np.sum(total_rewards))
    
    metrics = {
        # Glucose control
        "mean_glucose": mean_glucose,
        "time_in_range_percent": time_in_range,
        
        # Hypo metrics (separate steps and events)
        "hypo_steps": hypo_steps_count,
        "hypo_events": hypo_events_count,
        
        # Severe hyper metrics (separate steps and events)
        "severe_hyper_steps": severe_hyper_steps_count,
        "severe_hyper_events": severe_hyper_events_count,
        
        # Insulin accounting
        "total_bolus_units": total_bolus_units,
        "total_basal_units": total_basal_units,
        "total_insulin_units": total_insulin_units,
        
        # Reward breakdown
        "sum_zone_reward": sum_zone_reward,
        "sum_insulin_penalty": sum_insulin_penalty,
        "total_reward": total_reward,
    }
    
    # Add action counts
    metrics.update(action_counts)
    
    return metrics


class RewardShapingWrapper:
    """Wrapper for GlucoseEnv that optionally reshapes the zone reward for training experiments.

    This wrapper does NOT modify the simulator dynamics; it only replaces the per-step
    zone reward used for agent learning and logging. Original zone reward is preserved
    in `info['zone_reward_original']`.
    """

    def __init__(
        self,
        env: GlucoseEnv,
        enable: bool = False,
        in_range_reward: float | None = None,
        hypo_penalty: float | None = None,
        low_penalty: float | None = None,
        high_penalty: float | None = None,
        severe_penalty: float | None = None,
    ):
        self.env = env
        self.enable = enable
        # default to environment constants if not provided
        self.in_range_reward = (
            in_range_reward if in_range_reward is not None else float(env.REWARD_IN_RANGE)
        )
        self.hypo_penalty = hypo_penalty if hypo_penalty is not None else float(env.PENALTY_HYPO)
        self.low_penalty = low_penalty if low_penalty is not None else float(env.PENALTY_LOW)
        self.high_penalty = high_penalty if high_penalty is not None else float(env.PENALTY_HIGH)
        self.severe_penalty = (
            severe_penalty if severe_penalty is not None else float(env.PENALTY_SEVERE_HYPER)
        )

    # Delegate common attributes to the wrapped env
    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed: int | None = None, options: dict | None = None):
        return self.env.reset(seed=seed)

    def step(self, action: int):
        obs, orig_reward, terminated, truncated, info = self.env.step(action)

        # preserve original values
        info["zone_reward_original"] = info.get("zone_reward", 0.0)
        info["insulin_penalty_original"] = info.get("insulin_penalty", 0.0)

        if not self.enable:
            # nothing to change
            return obs, orig_reward, terminated, truncated, info

        g = info.get("glucose", float(self.env.glucose))
        # Recompute shaped zone reward using wrapper configuration
        if self.env.TIR_LOW <= g <= self.env.TIR_HIGH:
            zone_shaped = float(self.in_range_reward)
        elif g < self.env.glucose_hypo:
            zone_shaped = float(self.hypo_penalty)
        elif g < self.env.TIR_LOW:
            zone_shaped = float(self.low_penalty)
        elif g > self.env.glucose_severe_hyper:
            zone_shaped = float(self.severe_penalty)
        else:
            zone_shaped = float(self.high_penalty)

        # Keep the environment's insulin penalty (configurable via env.param)
        insulin_pen = info.get("insulin_penalty", 0.0)
        new_reward = zone_shaped + insulin_pen

        # Expose shaped values for logging
        info["zone_reward"] = zone_shaped
        info["insulin_penalty"] = insulin_pen
        info["reward_shaped"] = new_reward

        return obs, new_reward, terminated, truncated, info


class TrainingVariationWrapper:
    """Wrap env to randomize initial conditions per episode for robustness training.

    This does not modify dynamics; it only perturbs initial state variables to expose
    the agent to a wider distribution of starting conditions.
    """

    def __init__(self, env: GlucoseEnv, enable: bool = False, glucose_jitter: float = 10.0):
        self.env = env
        self.enable = enable
        self.glucose_jitter = glucose_jitter

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, meta = self.env.reset(seed=seed)
        if not self.enable:
            return obs, meta

        # Perturb initial glucose within a small jitter around safe median
        low = self.env.glucose_safe_low
        high = self.env.glucose_safe_high
        median = 0.5 * (low + high)
        perturb = self.env.rng.normal(0, self.glucose_jitter)
        new_glucose = float(np.clip(median + perturb, self.env.glucose_min, self.env.glucose_max))
        # set state variables to reflect perturbed start
        self.env.glucose = new_glucose
        self.env.glucose_prev = new_glucose
        # Randomize steps_since_last_bolus to simulate different histories
        self.env.steps_since_last_bolus = int(self.env.rng.randint(0, max(1, self.env.bolus_cooldown_steps * 3)))
        obs = self.env._get_observation()
        return obs, meta

    def step(self, action: int):
        return self.env.step(action)


def evaluate_agent(
    agent: DQNAgent,
    num_episodes: int = 5,
) -> Tuple[Dict, List]:
    """Evaluate agent with greedy policy (epsilon=0).
    
    Returns:
        (aggregated_metrics, detailed_episode_metrics)
    """
    episode_metrics = []
    
    for _ in range(num_episodes):
        obs, _ = agent.env.reset()
        # init history deque with repeated initial obs
        from collections import deque
        history = deque(maxlen=agent.history_len)
        for _ in range(agent.history_len):
            history.append(obs)
        state = np.concatenate(list(history))

        glucose_vals = []
        actions_vals = []
        hypo_steps = []
        hypo_events = []
        severe_hyper_steps = []
        severe_hyper_events = []
        bolus_vals = []
        zone_rewards = []
        insulin_penalties = []
        total_rewards = []
        
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0)

            step_result = agent.env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            glucose_vals.append(info["glucose"])
            actions_vals.append(action)
            hypo_steps.append(info["hypo_step"])
            hypo_events.append(info["hypo_event"])
            severe_hyper_steps.append(info["severe_hyper_step"])
            severe_hyper_events.append(info["severe_hyper_event"])
            bolus_vals.append(info["bolus_delivered"])
            zone_rewards.append(info["zone_reward"])
            insulin_penalties.append(info["insulin_penalty"])
            total_rewards.append(reward)
            
            # update history and stacked state
            history.append(next_state)
            next_stacked = np.concatenate(list(history))
            state = next_stacked
        
        metrics = compute_episode_metrics(
            glucose_vals, actions_vals,
            hypo_steps, hypo_events,
            severe_hyper_steps, severe_hyper_events,
            bolus_vals, zone_rewards, insulin_penalties, total_rewards,
            basal_rate=agent.env.basal_rate,
            bolus_actions=agent.env.bolus_actions,
            dt_minutes=agent.env.dt_minutes,
        )
        episode_metrics.append(metrics)
    
    # Aggregate
    aggregated = {}
    for key in episode_metrics[0].keys():
        values = [m[key] for m in episode_metrics]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    
    return aggregated, episode_metrics


def summarize_rows(rows: List[Dict]) -> Dict[str, float]:
    """Mean summary for a list of metric rows."""
    if not rows:
        return {}

    keys = [
        "time_in_range_percent",
        "hypo_steps",
        "hypo_events",
        "severe_hyper_steps",
        "severe_hyper_events",
        "total_bolus_units",
        "total_basal_units",
        "total_insulin_units",
        "sum_zone_reward",
        "sum_insulin_penalty",
        "total_reward",
        "action_0_count",
        "action_1_count",
        "action_2_count",
        "action_3_count",
        "action_4_count",
    ]
    summary = {}
    for k in keys:
        summary[k] = float(np.mean([r[k] for r in rows]))
    return summary


def print_sweep_summary(sweep_results: List[Dict]) -> None:
    """Print compact comparison across insulin penalty coefficients."""
    print("\nReward Coefficient Sweep Summary (means over eval episodes)")
    print(
        "coeff |  TIR% | hypo_s/e | severe_s/e | bolusU | basalU | totalU | "
        "A0 A1 A2 A3 A4 | zoneR | insPen | totalR"
    )
    for r in sweep_results:
        s = r["summary"]
        print(
            f"{r['coeff']:>5.2f} | "
            f"{s.get('time_in_range_percent', 0.0):5.1f} | "
            f"{s.get('hypo_steps', 0.0):5.1f}/{s.get('hypo_events', 0.0):4.1f} | "
            f"{s.get('severe_hyper_steps', 0.0):7.1f}/{s.get('severe_hyper_events', 0.0):4.1f} | "
            f"{s.get('total_bolus_units', 0.0):6.1f} | "
            f"{s.get('total_basal_units', 0.0):6.1f} | "
            f"{s.get('total_insulin_units', 0.0):6.1f} | "
            f"{s.get('action_0_count', 0.0):2.0f} "
            f"{s.get('action_1_count', 0.0):2.0f} "
            f"{s.get('action_2_count', 0.0):2.0f} "
            f"{s.get('action_3_count', 0.0):2.0f} "
            f"{s.get('action_4_count', 0.0):2.0f} | "
            f"{s.get('sum_zone_reward', 0.0):6.1f} | "
            f"{s.get('sum_insulin_penalty', 0.0):7.1f} | "
            f"{s.get('total_reward', 0.0):7.1f}"
        )


# ============================================================================
# Training Loop
# ============================================================================

def train(
    episodes: int = 1000,
    seed: int = 42,
    lr: float = 0.0005,
    gamma: float = 0.99,
    history_len: int = 6,
    replay_size: int = 10000,
    batch_size: int = 128,
    target_update_freq: int = 100,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: int = 500,
    eval_freq: int = 50,
    eval_episodes: int = 5,
    output_csv: str = "artifacts/dqn_training.csv",
    insulin_penalty_coeff: float = 0.25,
    min_replay_size: int = 1000,
    early_stop_patience: int | None = None,
    early_stop_min_evals: int = 4,
    safety_hypo_max: float = 3.0,
    safety_severe_max: float = 2.0,
    verbose: bool = False,
    save_models: bool = True,
    # Reward shaping for training-only experiments (wrapper, does not change dynamics)
    reward_shaping: bool = False,
    shape_in_range: float | None = None,
    shape_hypo: float | None = None,
    shape_low: float | None = None,
    shape_high: float | None = None,
    shape_severe: float | None = None,
    # Action smoothing (max discrete jump per step)
    max_action_jump: int = 1,
    # Training variability augmentation
    augment_variability: bool = False,
    variability_glucose_jitter: float = 10.0,
):
    """Train DQN agent."""
    
    # Deterministic seeds for reproducibility
    import random as _random
    np.random.seed(seed)
    _random.seed(seed)

    env = GlucoseEnv(
        random_seed=seed,
        insulin_penalty_coeff=insulin_penalty_coeff,
        verbose=False,
    )

    # Optionally wrap env for training-only reward shaping
    if reward_shaping:
        env = RewardShapingWrapper(
            env,
            enable=True,
            in_range_reward=shape_in_range,
            hypo_penalty=shape_hypo,
            low_penalty=shape_low,
            high_penalty=shape_high,
            severe_penalty=shape_severe,
        )

    if augment_variability:
        env = TrainingVariationWrapper(env, enable=True, glucose_jitter=variability_glucose_jitter)
    
    # If torch isn't available, skip DQN training gracefully so the
    # rest of the experiment pipeline (baselines, summaries) can run.
    if not TORCH_AVAILABLE:
        print("PyTorch is not installed. Skipping DQN training.\nInstall torch and re-run to enable DQN experiments.")
        return

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # extra deterministic settings for torch
    try:
        torch.manual_seed(seed)
        if device_str.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    agent = DQNAgent(
        history_len=history_len,
        env=env,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay,
        replay_buffer_size=replay_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        device=device_str,
        seed=seed,
    )
    # Apply action smoothing setting from train args
    agent.max_action_jump = max_action_jump
    
    print(f"Training DQN on GlucoseEnv")
    print(f"  Device: {agent.device}")
    print(f"  Global seed: {seed}")
    print(f"  Insulin penalty coeff: {insulin_penalty_coeff}")
    print(f"  Episodes: {episodes}")
    print(f"  Evaluation every {eval_freq} episodes ({eval_episodes} each)")
    if early_stop_patience is not None:
        print(
            f"  Early stopping: patience={early_stop_patience}, "
            f"min_evals={early_stop_min_evals}, "
            f"safety(hypo<={safety_hypo_max}, severe<={safety_severe_max})"
        )
    print()
    
    csv_rows = []

    # Best-checkpoint tracking with safety-aware ranking.
    # Best-checkpoint tracking with safety-aware ranking.
    # Rank format: (safety_feasible, tir, -hypo_events, reward)
    # prefer models that are safety-feasible, then higher TIR, then fewer hypo events, then reward
    best_rank = (-1.0, -np.inf, np.inf, -np.inf)
    best_train_episode = -1
    eval_rounds = 0
    no_improve_rounds = 0

    if output_csv:
        out_stem = os.path.splitext(os.path.basename(output_csv))[0]
    else:
        out_stem = f"dqn_coeff_{insulin_penalty_coeff:.2f}"
    best_model_path = os.path.join("artifacts", f"{out_stem}_best.pth")

    stop_training = False
    
    for episode in range(episodes):
        # Use episode-specific seed to get reproducible episodes
        ep_seed = seed + episode
        obs, _ = env.reset(seed=ep_seed)
        from collections import deque
        history = deque(maxlen=history_len)
        for _ in range(history_len):
            history.append(obs)
        state = np.concatenate(list(history))
        
        glucose_vals = []
        actions_vals = []
        hypo_steps = []
        hypo_events = []
        severe_hyper_steps = []
        severe_hyper_events = []
        bolus_vals = []
        zone_rewards = []
        insulin_penalties = []
        total_rewards = []
        losses = []
        
        done = False
        while not done:
            epsilon = agent.get_epsilon(episode)
            action = agent.select_action(state, epsilon=epsilon)
            
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            # update history with raw next_state then form stacked next state
            history.append(next_state)
            next_stacked = np.concatenate(list(history))

            agent.replay_buffer.push(state, action, reward, next_stacked, done)
            
            # require minimum replay buffer before training to avoid noisy early updates
            if len(agent.replay_buffer) >= batch_size and len(agent.replay_buffer) >= min_replay_size:
                loss = agent.train_step()
                losses.append(loss)
                agent.total_steps += 1
                
                if agent.total_steps % target_update_freq == 0:
                    agent.update_target_network()
            
            glucose_vals.append(info["glucose"])
            actions_vals.append(action)
            hypo_steps.append(info["hypo_step"])
            hypo_events.append(info["hypo_event"])
            severe_hyper_steps.append(info["severe_hyper_step"])
            severe_hyper_events.append(info["severe_hyper_event"])
            bolus_vals.append(info["bolus_delivered"])
            zone_rewards.append(info["zone_reward"])
            insulin_penalties.append(info["insulin_penalty"])
            total_rewards.append(reward)
            
            state = next_stacked

        # Per-episode training metrics log.
        train_metrics = compute_episode_metrics(
            glucose_vals, actions_vals,
            hypo_steps, hypo_events,
            severe_hyper_steps, severe_hyper_events,
            bolus_vals, zone_rewards, insulin_penalties, total_rewards,
            basal_rate=env.basal_rate,
            bolus_actions=env.bolus_actions,
            dt_minutes=env.dt_minutes,
        )
        train_row = {
            "phase": "train",
            "coeff": insulin_penalty_coeff,
            "train_episode": episode,
            "eval_episode": -1,
            "epsilon": agent.get_epsilon(episode),
        }
        train_row.update(train_metrics)
        csv_rows.append(train_row)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_agg, eval_episodes_detail = evaluate_agent(agent, num_episodes=eval_episodes)
            eval_rounds += 1
            
            for eval_ep, metrics in enumerate(eval_episodes_detail):
                row = {
                    "phase": "eval",
                    "coeff": insulin_penalty_coeff,
                    "train_episode": episode,
                    "eval_episode": eval_ep,
                    "epsilon": 0.0,
                }
                row.update(metrics)
                csv_rows.append(row)
            
            if verbose or (episode + 1) % (eval_freq * 4) == 0:
                tir = eval_agg["time_in_range_percent"]["mean"]
                reward = eval_agg["total_reward"]["mean"]
                insulin = eval_agg["total_insulin_units"]["mean"]
                print(
                    f"[Episode {episode+1:4d}] eps={agent.get_epsilon(episode):.3f} | "
                    f"Eval TIR={tir:5.1f}% | Insulin={insulin:6.0f}U | Reward={reward:7.0f}"
                )

            # Safety-aware best model selection.
            current_tir = eval_agg["time_in_range_percent"]["mean"]
            current_reward = eval_agg["total_reward"]["mean"]
            current_hypo = eval_agg["hypo_events"]["mean"]
            current_severe = eval_agg["severe_hyper_events"]["mean"]
            safety_feasible = float(
                (current_hypo <= safety_hypo_max) and (current_severe <= safety_severe_max)
            )
            current_rank = (safety_feasible, current_tir, -float(current_hypo), current_reward)

            if current_rank > best_rank:
                best_rank = current_rank
                best_train_episode = episode
                no_improve_rounds = 0
                if save_models:
                    os.makedirs("artifacts", exist_ok=True)
                    torch.save(agent.q_net.state_dict(), best_model_path)
                    print(
                        f"  New best checkpoint @ ep {episode+1}: "
                        f"TIR={current_tir:.2f}, reward={current_reward:.1f}, "
                        f"hypo={current_hypo:.2f}, severe={current_severe:.2f}"
                    )
            else:
                no_improve_rounds += 1

            # Early stopping on no improvement in best-rank objective.
            if (
                early_stop_patience is not None
                and eval_rounds >= early_stop_min_evals
                and no_improve_rounds >= early_stop_patience
            ):
                print(
                    f"Early stopping triggered at episode {episode+1} "
                    f"(no improvement for {no_improve_rounds} eval rounds)."
                )
                stop_training = True

        if stop_training:
            break
    
    print("\nTraining complete!")
    
    if save_models:
        os.makedirs("artifacts", exist_ok=True)
        torch.save(agent.q_net.state_dict(), "artifacts/dqn_q_net.pth")
        print(f"  Model saved to artifacts/dqn_q_net.pth")
        if best_train_episode >= 0:
            print(
                f"  Best model saved to {best_model_path} "
                f"(train episode {best_train_episode + 1})"
            )
    
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        if csv_rows:
            with open(output_csv, "w", newline="") as f:
                fieldnames = list(csv_rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
        print(f"  Logs saved to {output_csv}")

    # Compact run summary from evaluation rows.
    eval_rows = [r for r in csv_rows if r["phase"] == "eval"]
    run_summary = summarize_rows(eval_rows) if eval_rows else {}
    return agent, csv_rows, run_summary


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on GlucoseEnv")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--replay-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=350)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--out", type=str, default="artifacts/dqn_training.csv")
    parser.add_argument("--history-len", type=int, default=6,
                        help="Number of past observations to stack for state input")
    parser.add_argument("--insulin-penalty-coeff", type=float, default=0.25)
    parser.add_argument("--reward-shaping", action="store_true",
                        help="Enable training-only reward shaping via wrapper")
    parser.add_argument("--shape-in-range", type=float, default=None,
                        help="Shaped in-range reward (overrides env default when set)")
    parser.add_argument("--shape-hypo", type=float, default=None,
                        help="Shaped hypo penalty")
    parser.add_argument("--shape-low", type=float, default=None,
                        help="Shaped penalty for below-TIR but above-hypo")
    parser.add_argument("--shape-high", type=float, default=None,
                        help="Shaped penalty for above-TIR but below-severe")
    parser.add_argument("--shape-severe", type=float, default=None,
                        help="Shaped severe-hyper penalty")
    parser.add_argument("--max-action-jump", type=int, default=1,
                        help="Max discrete action index change permitted between steps")
    parser.add_argument("--augment-variability", action="store_true",
                        help="Randomize initial glucose/bolus-timing per episode during training")
    parser.add_argument("--variability-glucose-jitter", type=float, default=10.0,
                        help="Std dev (mg/dL) for initial glucose perturbation when augmenting variability")
    parser.add_argument("--min-replay-size", type=int, default=1000,
                        help="Minimum replay buffer size before training starts")
    parser.add_argument("--sweep-insulin-penalty", action="store_true")
    parser.add_argument("--sweep-coeffs", type=str, default="0.25,0.35,0.5")
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-evals", type=int, default=4)
    parser.add_argument("--safety-hypo-max", type=float, default=3.0)
    parser.add_argument("--safety-severe-max", type=float, default=2.0)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    output_csv = None if args.no_csv else args.out

    if args.sweep_insulin_penalty:
        coeffs = [float(x.strip()) for x in args.sweep_coeffs.split(",") if x.strip()]
        sweep_results = []

        for coeff in coeffs:
            run_out = None
            if output_csv:
                root, ext = os.path.splitext(output_csv)
                run_out = f"{root}_coeff_{coeff:.2f}{ext or '.csv'}"

            _, _, run_summary = train(
                episodes=args.episodes,
                seed=args.seed,
                lr=args.lr,
                gamma=args.gamma,
                history_len=args.history_len,
                min_replay_size=args.min_replay_size,
                replay_size=args.replay_size,
                batch_size=args.batch_size,
                target_update_freq=args.target_update_freq,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay=args.epsilon_decay,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                output_csv=run_out,
                insulin_penalty_coeff=coeff,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_evals=args.early_stop_min_evals,
                safety_hypo_max=args.safety_hypo_max,
                safety_severe_max=args.safety_severe_max,
                verbose=args.verbose,
                reward_shaping=args.reward_shaping,
                shape_in_range=args.shape_in_range,
                shape_hypo=args.shape_hypo,
                shape_low=args.shape_low,
                shape_high=args.shape_high,
                shape_severe=args.shape_severe,
                max_action_jump=args.max_action_jump,
                augment_variability=args.augment_variability,
                variability_glucose_jitter=args.variability_glucose_jitter,
            )
            sweep_results.append({"coeff": coeff, "summary": run_summary})

        print_sweep_summary(sweep_results)
    else:
        train(
            episodes=args.episodes,
            seed=args.seed,
            lr=args.lr,
            gamma=args.gamma,
            history_len=args.history_len,
            min_replay_size=args.min_replay_size,
            replay_size=args.replay_size,
            batch_size=args.batch_size,
            target_update_freq=args.target_update_freq,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            output_csv=output_csv,
            insulin_penalty_coeff=args.insulin_penalty_coeff,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_evals=args.early_stop_min_evals,
            safety_hypo_max=args.safety_hypo_max,
            safety_severe_max=args.safety_severe_max,
            verbose=args.verbose,
            reward_shaping=args.reward_shaping,
            shape_in_range=args.shape_in_range,
            shape_hypo=args.shape_hypo,
            shape_low=args.shape_low,
            shape_high=args.shape_high,
            shape_severe=args.shape_severe,
            max_action_jump=args.max_action_jump,
            augment_variability=args.augment_variability,
            variability_glucose_jitter=args.variability_glucose_jitter,
        )
        
