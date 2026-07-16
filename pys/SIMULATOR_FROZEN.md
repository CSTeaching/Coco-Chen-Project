Simulator frozen: 2026-05-07

This simulator version is the frozen baseline for final experiments.

Changes included before freezing:
- Meal absorption over 24 steps (2 hours)
- Insulin-on-board triangular absorption over 36 steps (3 hours)
- Removed double dt-scaling bug in bolus effect
- Baseline cooldown (120 min) added to baseline evaluators
- Validation suite added: baselines/validate_realism.py

Do not modify `glucose_env.py` or `data/plots_eda/eda_derived_simulator_params.json` while running final multi-seed experiments.
