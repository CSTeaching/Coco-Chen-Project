Setup instructions for running experiments locally

1) Create and activate a venv (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2) Install dependencies. For machines without GPU/CUDA, install the CPU build of PyTorch:

```bash
# Install common deps
pip install -r requirements.txt

# If the above fails for torch, install CPU-specific wheel:
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

3) Run tests and baseline evaluation:

```bash
python test_absorption_dynamics.py
python agents/train_dqn.py --episodes 10 --seed 42
```

Notes:
- `SIMULATOR_FROZEN.md` indicates the simulator must not be modified for final experiments.
- If you cannot install `torch`, DQN training will be skipped and baselines will still run; see logs for details.
