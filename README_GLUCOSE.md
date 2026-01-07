# Blood Glucose Management Reinforcement Learning Project

This project extends the fever-management RL framework to tackle **diabetes glucose control**. An RL agent learns to administer insulin doses to keep a patient's blood glucose within a safe, healthy range.

## Overview

### Problem
Type 2 diabetes patients struggle with blood glucose regulation. Too high (hyperglycemia) or too low (hypoglycemia) glucose levels are dangerous. Manual insulin dosing requires constant monitoring and adjustment. An RL agent could learn personalized, adaptive dosing strategies.

### Solution
- **Environment (`env/glucose_env.py`)**: Simulates realistic glucose dynamics including insulin pharmacokinetics, meal intake, and stochastic metabolism
- **Agent (`agents/train_glucose_q_learning.py`)**: Q-learning agent that learns a policy to dose insulin optimally
- **Evaluation (`agents/eval_glucose_q_learning.py`, `agents/eval_glucose_visual.py`)**: Assess agent performance on glucose control metrics

## Environment Details

### State
- **Glucose level (mg/dL)**: Continuous value, range 40–400
- Observation: `[glucose]` as a 1-D numpy array

### Actions
**Discrete insulin doses (6 actions):**
- Action 0: 0 units (no insulin)
- Action 1: 2 units
- Action 2: 4 units
- Action 3: 6 units
- Action 4: 8 units
- Action 5: 10 units

### Glucose Dynamics

Each step simulates approximately **5 minutes** of real-time glucose change:

1. **Insulin effect**: Decreases glucose proportional to dose and insulin sensitivity
   - Insulin sensitivity varies by patient type (more insulin-resistant patients have lower sensitivity)
   
2. **Basal hepatic glucose production**: Gluconeogenesis and glycogenolysis raise glucose naturally
   
3. **Meal intake**: Stochastic carbohydrate intake increases glucose (occurs with configurable probability)
   
4. **Natural metabolism**: Glucose utilization by tissues and excretion decays glucose
   
5. **Stochastic variability**: Realistic metabolic variability (±0–15% depending on patient type)

### Patient Types

Three patient models with different insulin sensitivities and meal variability:

- **`type2_stable`**: Predictable insulin response, consistent meals
  - Insulin sensitivity: 15 mg/dL per unit
  - Meal variability: moderate
  
- **`type2_variable`**: Less predictable response, variable meals
  - Insulin sensitivity: 12 mg/dL per unit
  - Meal variability: high
  
- **`brittle_diabetes`**: Highly unpredictable dynamics
  - Insulin sensitivity: 10 mg/dL per unit
  - Meal variability: very high

### Reward Shaping

**Safe glucose range:** 80–180 mg/dL (standard diabetes management target)

**Reward modes:**

1. **`standard`**: Balanced incentives
   - Base penalty for distance from target (120 mg/dL)
   - Penalty for hypoglycemia (<80), hyperglycemia (>180)
   - Penalty for large glucose swings

2. **`aggressive_reward`**: Strict adherence to range
   - Heavy penalties for out-of-range glucose
   - Bonus for staying in range
   - Suitable for tightly-controlled scenarios

3. **`conservative_reward`**: Gentle incentives
   - Small penalties for out-of-range
   - Penalty for high insulin doses (encourages minimal intervention)
   - Suitable for learning cautious dosing

4. **`variable_meals`**: Adapted for meal variability
   - Similar to standard but tuned for unpredictable meals

### Termination

Episodes end when:
- **Hypoglycemia**: glucose < 70 mg/dL (dangerous low) → heavy penalty
- **Hyperglycemia**: glucose > 250 mg/dL (dangerous high) → heavy penalty
- **Max steps**: 288 steps (24 hours at 5-min intervals) reached

## Scripts

### Training

```bash
python agents/train_glucose_q_learning.py \
  --episodes 2000 \
  --bins 30 \
  --mode standard \
  --patient type2_stable \
  --seed 0
```

**Output:** `glucose_q_table_{mode}_{patient}.npz` (saved Q-table and bin edges)

**Arguments:**
- `--episodes`: Number of training episodes (default: 2000)
- `--bins`: Glucose discretization bins (default: 30)
- `--mode`: Reward mode (default: `standard`)
- `--patient`: Patient type (default: `type2_stable`)
- `--seed`: Random seed (default: 0)

### Evaluation

```bash
python agents/eval_glucose_q_learning.py \
  --q_table glucose_q_table_standard_type2_stable.npz \
  --episodes 5
```

**Output:** Console traces showing glucose, insulin, and reward at each step; aggregate metrics (avg glucose, min/max, time in range %, total reward)

### Visual Evaluation

```bash
python agents/eval_glucose_visual.py \
  --q_table glucose_q_table_standard_type2_stable.npz \
  --episodes 10
```

**Output PNGs:**
- `glucose_traces.png` – Glucose over time for first 5 episodes
- `insulin_actions.png` – Insulin doses over time
- `reward_traces.png` – Reward traces
- `glucose_metrics.png` – Histograms and scatter plots of key metrics

**Metrics printed:**
- Average glucose (mean ± std)
- Average min/max glucose
- Average time in range (80–180 mg/dL)
- Average total reward
- Average episode length

## Quick Start

1. **Create and activate virtualenv:**
   ```bash
   python3 -m venv .venv_glucose
   source .venv_glucose/bin/activate
   pip install gym numpy matplotlib pandas
   ```

2. **Train an agent:**
   ```bash
   cd /workspaces/Project
   python agents/train_glucose_q_learning.py --episodes 1000 --mode standard --patient type2_stable
   ```

3. **Evaluate and visualize:**
   ```bash
   python agents/eval_glucose_q_learning.py --q_table glucose_q_table_standard_type2_stable.npz --episodes 5
   python agents/eval_glucose_visual.py --q_table glucose_q_table_standard_type2_stable.npz --episodes 10
   ```

## Results & Insights

### Typical Agent Performance (after 500 episodes training)

| Metric | Value |
|--------|-------|
| Avg glucose | ~150 mg/dL |
| Time in range (80–180) | ~70–80% |
| Min glucose | ~100 mg/dL (safe) |
| Max glucose | ~250–280 mg/dL (occasional danger) |

**Interpretation:**
- Agent learns to keep glucose reasonably controlled (~70–80% in safe range)
- Occasionally insulin doses cause glucose to drift high (>250), triggering penalty
- Longer training helps refine dosing strategy to avoid hyperglycemic episodes
- Different reward modes and patient types produce different learned behaviors

### Investigations

The framework allows exploration of:

1. **Reward structure effects**: How aggressive vs. conservative rewards affect learned behavior
2. **Patient variability**: How different patient types (stable vs. brittle) require different agent strategies
3. **Action granularity**: More/fewer insulin dose options and their impact on control
4. **Training duration**: Convergence speed and final performance quality

## Future Enhancements

- Multi-step planning (use actor-critic, DQN, or policy gradient methods)
- Meal predictions or sensors (if available in real data)
- Multi-agent scenarios (patient + doctor + agent collaboration)
- Integration with real VitalDB glucose data for validation
- Continuous action space for fine-grained insulin titration
- Safety constraints (hard bounds on insulin dose changes per step)

## References

- Glucose pharmacokinetics: simplified Bergman minimal model
- Standard diabetes targets: ADA clinical practice guidelines (80–130 mg/dL fasting, <180 post-meal)
- RL framework: OpenAI Gym API, Q-learning
