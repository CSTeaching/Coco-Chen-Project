"""Blood Glucose Management Environment using Gym API.

This environment simulates glucose dynamics in a type 2 diabetes patient.
An agent chooses insulin doses at each timestep to keep glucose in a healthy range.

Glucose dynamics are based on simplified pharmacokinetic models:
- Glucose increases due to carbohydrate intake
- Glucose decreases due to insulin action and natural metabolism
- Stochastic variability in metabolism and meal timing

State: glucose level (mg/dL), range ~70-400
Actions: insulin dose (units), discrete choices 0-5
  - 0: no insulin
  - 1: 2 units
  - 2: 4 units
  - 3: 6 units
  - 4: 8 units
  - 5: 10 units

Reward: encourages glucose 80-180 mg/dL, penalizes dangerous extremes
"""

import numpy as np
import gym
from gym import spaces


class GlucoseEnv(gym.Env):
    """Blood glucose management environment.
    
    Observation: glucose level (mg/dL)
    Action: insulin dose (units)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, seed=None, mode='standard', patient_type='type2_stable'):
        """Initialize the glucose environment.
        
        Args:
            seed: random seed for reproducibility
            mode: 'standard', 'aggressive_reward', 'conservative_reward', 'variable_meals'
            patient_type: 'type2_stable', 'type2_variable', 'brittle_diabetes'
        """
        super().__init__()

        # Action space: insulin dose in units (0, 2, 4, 6, 8, 10)
        self.action_space = spaces.Discrete(6)
        # Observation space: glucose level mg/dL (typically 40-400)
        self.observation_space = spaces.Box(
            low=np.array([40.0], dtype=np.float32),
            high=np.array([400.0], dtype=np.float32),
            dtype=np.float32
        )

        self._rng = np.random.RandomState(seed)
        self.seed_val = seed
        self.mode = mode
        self.patient_type = patient_type
        
        # Validate inputs
        if mode not in ('standard', 'aggressive_reward', 'conservative_reward', 'variable_meals'):
            raise ValueError(f"Unknown mode: {mode}")
        if patient_type not in ('type2_stable', 'type2_variable', 'brittle_diabetes'):
            raise ValueError(f"Unknown patient_type: {patient_type}")

        # Glucose dynamics parameters (vary by patient type)
        self._set_patient_parameters()

        # Time step counter for meal scheduling
        self.step_count = 0
        self.reset()

    def _set_patient_parameters(self):
        """Set glucose dynamics parameters based on patient type."""
        if self.patient_type == 'type2_stable':
            # Stable type 2 diabetic: relatively predictable insulin sensitivity
            self.insulin_sensitivity = 15.0  # mg/dL per unit insulin
            self.basal_glucose_production = 1.5  # mg/dL/step increase without insulin
            self.glucose_decay = 0.98  # natural glucose decay per step
            self.meal_carbs_std = 40.0  # std dev of meal size
            self.meal_prob = 0.15  # probability of meal at each step
            self.metabolism_std = 0.05  # stochasticity in metabolism
        elif self.patient_type == 'type2_variable':
            # Variable type 2 diabetic: less predictable response
            self.insulin_sensitivity = 12.0  # lower sensitivity (less responsive)
            self.basal_glucose_production = 2.0
            self.glucose_decay = 0.97
            self.meal_carbs_std = 60.0  # more variable meals
            self.meal_prob = 0.18
            self.metabolism_std = 0.10  # higher stochasticity
        else:  # brittle_diabetes
            # Brittle diabetes: highly unpredictable
            self.insulin_sensitivity = 10.0
            self.basal_glucose_production = 2.5
            self.glucose_decay = 0.96
            self.meal_carbs_std = 80.0
            self.meal_prob = 0.20
            self.metabolism_std = 0.15

        # Safe glucose range: 80-180 mg/dL (typical target for adults)
        self.glucose_min_safe = 80.0
        self.glucose_max_safe = 180.0
        # Critical thresholds
        self.glucose_min_critical = 70.0  # hypoglycemia danger
        self.glucose_max_critical = 250.0  # hyperglycemia danger

    def reset(self):
        """Start a new episode.
        
        Initial glucose is typically fasting: 100-130 mg/dL for diabetics.
        """
        self.glucose = float(self._rng.uniform(100.0, 130.0))
        self.done = False
        self.step_count = 0
        return np.array([self.glucose], dtype=np.float32)

    def step(self, action):
        """Apply insulin dose, simulate glucose dynamics.
        
        Returns: (obs, reward, done, info)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Map action to insulin dose (units)
        insulin_dose = float(action * 2)  # 0, 2, 4, 6, 8, 10 units

        # Track previous glucose for reward shaping
        prev_glucose = self.glucose

        # --- Glucose dynamics ---
        # 1. Insulin effect: decreases glucose
        glucose_delta = -self.insulin_sensitivity * insulin_dose / 100.0

        # 2. Basal glucose production (from liver, etc.): increases glucose
        glucose_delta += self.basal_glucose_production

        # 3. Natural glucose decay (metabolism, diffusion)
        glucose_delta += (self.glucose_decay - 1.0) * self.glucose

        # 4. Stochastic meal intake
        # In variable_meals mode, meals occur more randomly; otherwise scheduled times
        if self.mode == 'variable_meals' or self._rng.rand() < self.meal_prob:
            # Carb intake: absorbed gradually, raises glucose
            meal_carbs = max(0, self._rng.normal(40.0, self.meal_carbs_std))
            # Convert carbs to glucose rise (rough: ~5 mg/dL per 10g carbs)
            glucose_delta += meal_carbs * 0.5

        # 5. Stochastic metabolism factor
        metabolism_factor = 1.0 + self._rng.normal(0, self.metabolism_std)
        glucose_delta *= metabolism_factor

        # Apply delta
        self.glucose = float(self.glucose + glucose_delta)

        # Cap glucose within physiological bounds (can't go below ~40 or above ~500)
        self.glucose = np.clip(self.glucose, 40.0, 500.0)

        # --- Reward calculation ---
        reward = self._compute_reward(prev_glucose, insulin_dose)

        # --- Termination conditions ---
        info = {}
        if self.glucose < self.glucose_min_critical or self.glucose > self.glucose_max_critical:
            self.done = True
            info['termination'] = 'danger'
            if self.glucose < self.glucose_min_critical:
                info['danger_type'] = 'hypoglycemia'
                reward -= 20.0  # severe penalty
            else:
                info['danger_type'] = 'hyperglycemia'
                reward -= 15.0  # severe penalty
        elif self.glucose_min_safe <= self.glucose <= self.glucose_max_safe:
            # Safe range: could give bonus, but don't terminate
            reward += 2.0
            self.done = False
        else:
            self.done = False

        self.step_count += 1
        # Limit episode length to 288 steps (24 hours at 5-min intervals)
        if self.step_count >= 288:
            self.done = True
            info['termination'] = 'max_steps'

        obs = np.array([self.glucose], dtype=np.float32)
        return obs, float(reward), bool(self.done), info

    def _compute_reward(self, prev_glucose, insulin_dose):
        """Compute reward based on glucose level and mode."""
        # Base reward: penalty for distance from target
        target_glucose = 120.0
        distance = abs(self.glucose - target_glucose)
        reward = -distance * 0.05  # smaller penalty for small deviations

        if self.mode == 'standard':
            # Modest rewards/penalties
            if self.glucose < self.glucose_min_safe:
                reward -= 5.0  # hypoglycemia penalty
            elif self.glucose > self.glucose_max_safe:
                reward -= 3.0  # hyperglycemia penalty
            # Reward for stable glucose (small change)
            change = abs(self.glucose - prev_glucose)
            reward += -0.1 * change

        elif self.mode == 'aggressive_reward':
            # Strict rewards: heavily penalize out-of-range
            if self.glucose < self.glucose_min_safe:
                reward -= 10.0
            elif self.glucose > self.glucose_max_safe:
                reward -= 8.0
            # Bonus for staying in range
            if self.glucose_min_safe <= self.glucose <= self.glucose_max_safe:
                reward += 5.0

        elif self.mode == 'conservative_reward':
            # Gentle rewards: avoid extreme dosing
            if self.glucose < self.glucose_min_safe:
                reward -= 2.0
            elif self.glucose > self.glucose_max_safe:
                reward -= 1.5
            # Penalty for high insulin doses (encourage minimal intervention)
            if insulin_dose > 4.0:
                reward -= 0.5

        elif self.mode == 'variable_meals':
            # Similar to standard but adapt to meal variability
            if self.glucose < self.glucose_min_safe:
                reward -= 5.0
            elif self.glucose > self.glucose_max_safe:
                reward -= 3.0

        return reward

    def render(self, mode='human'):
        print(f"Glucose: {self.glucose:.1f} mg/dL | Step: {self.step_count}")

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)
        self.seed_val = seed
