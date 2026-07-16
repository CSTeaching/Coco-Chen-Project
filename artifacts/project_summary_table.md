# Project Summary Table

Below is a concise table summarizing the main components, locations, purposes, and status of items in this workspace.

| Component | Path | Purpose | Status | Notes |
|---|---|---|---|---|
| Simulator core | `glucose_env.py` | Gym-compatible glucose simulator; dynamics, reward, obs/action spaces | Done | v2 dynamics: meal absorption (24 steps), insulin IOB (36 steps); insulin potency bug fixed. |
| DQN training | `agents/train_dqn.py` | DQN training loop with evaluation, early stopping, checkpointing | Done | Best-checkpoint selection implemented; smoke tests passed. |
| Multi-seed runner | `multi_seed_runner.py` | Orchestrates multi-seed experiments and aggregates CSVs | Done | Produced `artifacts/final_dqn_aggregated.csv` and per-seed CSVs. |
| Baselines | `baselines/eval_baselines.py` | Baseline policies (NoInsulin, SingleThreshold, TwoThreshold) | Done | Added bolus cooldown and reset hooks for realism. |
| Realism validator | `baselines/validate_realism.py` | Automated checks (insulin totals, meal peaks, extreme-zone time) | Done | Outputs `artifacts/realism_policy_metrics.csv`, `artifacts/realism_validation_summary.csv`. Passing. |
| Tests | `test_absorption_dynamics.py` | Unit-style checks for meal/insulin absorption dynamics | Done | Validates temporal absorption behavior. |
| Simulator freeze marker | `SIMULATOR_FREEZE.md` | Specifies simulator version used for final experiments | Done | Included for reproducibility. |
| Artifacts (aggregated) | `artifacts/final_dqn_aggregated.csv` | Aggregated multi-seed training/eval CSV | Done | Contains per-episode seed-merged metrics (TIR, hypo events, insulin totals). |
| Artifacts (per-seed) | `artifacts/final_dqn_seed*.csv` | Per-seed training/eval logs | Done | Seeds: 42,123,456,789,999 present. |
| Best-model checkpoints | `artifacts/*.pth` | Saved best DQN models per seed | Done | Per-seed best `.pth` files available. |
| EDA params | `data/plots_eda/eda_derived_simulator_params.json` | Data-driven simulator params (basal, meal sizes, sensitivities) | Present | Basal ≈0.978 U/hr; median bolus ≈4.8U; meal median ≈36g. |
| Prototype scripts | `prototype stuff/` | Experiments, visualizers, toy envs | Mixed | Contains helper scripts and Q-tables for toy experiments. |
| Requirements | `prototype stuff/requirements.txt` | Python dependencies for prototype code | Present | PyTorch installed in venv for experiments. |
| Plots | `plots/` & `plots/multi_seed/` | Target location for learning curves and figures | Pending | Not yet generated for final aggregated CSV (next step). |
| Final deliverables | `artifacts/` + `SIMULATOR_FREEZE.md` | Bundle for submission (figures, aggregated CSVs, freeze doc) | Pending | Packaging/readme to prepare. |

---

If you want, I can now:
- Generate plots from `artifacts/final_dqn_aggregated.csv` and save to `plots/multi_seed/`.
- Produce a seed-averaged CSV/table of final metrics (TIR, hypo events, insulin usage).
- Create a final README and package artifacts into a zip.

Choose one and I'll proceed.