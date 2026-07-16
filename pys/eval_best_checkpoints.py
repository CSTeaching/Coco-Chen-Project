from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from agents.train_dqn import DQNAgent, GlucoseEnv, evaluate_agent


DEFAULT_SEEDS = [42, 123, 456, 789, 999]


def infer_history_len(checkpoint_path: Path) -> int:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    first_weight = state_dict["net.0.weight"]
    input_dim = int(first_weight.shape[1])
    if input_dim % 5 != 0:
        raise ValueError(f"Unexpected input dimension {input_dim} in {checkpoint_path.name}")
    return input_dim // 5


def load_agent(checkpoint_path: Path, seed: int, history_len: int, insulin_penalty_coeff: float) -> DQNAgent:
    env = GlucoseEnv(random_seed=seed, insulin_penalty_coeff=insulin_penalty_coeff, verbose=False)
    agent = DQNAgent(
        env=env,
        history_len=history_len,
        lr=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=2000,
        replay_buffer_size=10000,
        batch_size=128,
        target_update_freq=100,
        device="cpu",
        seed=seed,
    )
    state_dict = torch.load(checkpoint_path, map_location=agent.device)
    agent.q_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    agent.q_net.eval()
    agent.target_net.eval()
    return agent


def summarize_checkpoint(checkpoint_path: Path, seed: int, eval_episodes: int, history_len: int, insulin_penalty_coeff: float) -> dict:
    agent = load_agent(checkpoint_path, seed, history_len, insulin_penalty_coeff)
    eval_agg, _ = evaluate_agent(agent, num_episodes=eval_episodes)
    tir_mean = eval_agg["time_in_range_percent"]["mean"]
    hypo_mean = eval_agg["hypo_events"]["mean"]
    severe_mean = eval_agg["severe_hyper_events"]["mean"]
    return {
        "seed": seed,
        "checkpoint": checkpoint_path.name,
        "tir_mean": tir_mean,
        "hypo_mean": hypo_mean,
        "severe_mean": severe_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best DQN checkpoints across seeds")
    parser.add_argument("--checkpoint-prefix", type=str, default="final_dqn")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--history-len", type=int, default=0,
                        help="Override inferred history length; 0 means auto-detect")
    parser.add_argument("--insulin-penalty-coeff", type=float, default=0.25)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    artifact_dir = root / "artifacts"
    rows = []
    for seed in args.seeds:
        checkpoint = artifact_dir / f"{args.checkpoint_prefix}_seed{seed}_best.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        inferred_history_len = infer_history_len(checkpoint)
        rows.append(
            summarize_checkpoint(
                checkpoint,
                seed,
                args.eval_episodes,
                inferred_history_len if args.history_len <= 0 else args.history_len,
                args.insulin_penalty_coeff,
            )
        )

    tirs = [row["tir_mean"] for row in rows]
    hypos = [row["hypo_mean"] for row in rows]
    severes = [row["severe_mean"] for row in rows]

    print("Best checkpoint evaluation summary")
    for row in rows:
        print(
            f"seed={row['seed']}: TIR={row['tir_mean']:.2f}%, "
            f"hypo_events={row['hypo_mean']:.2f}, severe_hyper_events={row['severe_mean']:.2f}"
        )
    print(
        f"AVERAGE: TIR={mean(tirs):.2f}% ± {pstdev(tirs):.2f}, "
        f"hypo_events={mean(hypos):.2f} ± {pstdev(hypos):.2f}, "
        f"severe_hyper_events={mean(severes):.2f} ± {pstdev(severes):.2f}"
    )


if __name__ == "__main__":
    main()
