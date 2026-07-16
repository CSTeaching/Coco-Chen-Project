Simulator freeze

This repository's simulator was frozen on 2026-04-27 after the following changes:

- Replaced instant meal and insulin impulses with temporal absorption curves.
- Meal absorption: uniform over 24 steps (2 hours).
- Insulin absorption: triangular over 36 steps (3 hours), peak near 1 hour.
- Removed double time-scaling of absorbed insulin in `glucose_env.py`.
- Added baseline cooldown in `baselines/eval_baselines.py`.

Do not modify `glucose_env.py` or the EDA parameter file while running multi-seed experiments.

Tag: simulator_v1_frozen_2026-04-27
