"""Toy fever simulator Gym environment.

State:
 - temperature (float) in °C, range roughly [36, 41]

Actions (Discrete(4)):
 - 0: no medicine -> temp +0.1°C
 - 1: small dose   -> temp -1.0°C
 - 2: medium dose  -> temp -1.5°C (10% side effect: +1°C)
 - 3: large dose   -> temp -2.0°C (30% side effect: +1°C)

Episode termination:
 - temperature < 36.0 or > 41.0 -> danger (done)
 - temperature in [36.9, 37.1] -> recovered (done)

Reset starts at 39.0°C.
"""

import numpy as np
import gym
from gym import spaces


class FeverEnv(gym.Env):
    """A simple toy fever simulator compatible with OpenAI Gym's classic API.

    Observation: np.array([temperature], dtype=np.float32)
    Action: discrete 0..3
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([36.0], dtype=np.float32),
                                            high=np.array([41.0], dtype=np.float32),
                                            dtype=np.float32)
        # use a RandomState for deterministic behavior when seed provided
        self._rng = np.random.RandomState(seed)
        self.seed_val = seed
        self.reset()

    def reset(self):
        """Start a new episode.

        Changes from previous simple env:
        - Starting temperature is randomized in [38.0, 40.0] to vary difficulty.
        Returns observation as a 1-D numpy array containing the temperature.
        """
        # Randomize starting temperature between 38 and 40°C for realism.
        self.temp = float(self._rng.uniform(38.0, 40.0))
        self.done = False
        return np.array([self.temp], dtype=np.float32)

    def step(self, action: int):
        """Apply `action`, update temperature, compute reward and done flag.

        Returns: (obs, reward, done, info)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")


        # --- New, more realistic medicine/stochastic effects ---
        # action mapping:
        # 0: no medicine (small natural increase)
        # 1: small dose  -> decrease by uniform(0.8, 1.2)
        # 2: medium dose -> decrease by uniform(1.3, 1.7)
        # 3: large dose  -> decrease by uniform(1.8, 2.2)
        # Each dose also has a chance to cause a side-effect which increases
        # temperature by uniform(0.5, 1.0). Probabilities:
        # small: 5%, medium: 10%, large: 20%.

        side_effect = False
        side_effect_mag = 0.0

        if action == 0:
            # no medicine: slight natural increase
            delta = 0.1
        elif action == 1:
            # small dose: random decrease in [0.8, 1.2]
            delta = -float(self._rng.uniform(0.8, 1.2))
            if self._rng.rand() < 0.05:
                side_effect = True
                side_effect_mag = float(self._rng.uniform(0.5, 1.0))
        elif action == 2:
            # medium dose: random decrease in [1.3, 1.7]
            delta = -float(self._rng.uniform(1.3, 1.7))
            if self._rng.rand() < 0.10:
                side_effect = True
                side_effect_mag = float(self._rng.uniform(0.5, 1.0))
        elif action == 3:
            # large dose: random decrease in [1.8, 2.2]
            delta = -float(self._rng.uniform(1.8, 2.2))
            if self._rng.rand() < 0.20:
                side_effect = True
                side_effect_mag = float(self._rng.uniform(0.5, 1.0))
        else:
            delta = 0.0

        # If a side-effect occurs, add its warming effect
        if side_effect:
            delta += side_effect_mag

        # Cap behavior: no single step may drop temp below 36.0°C.
        # If the computed delta would take temp below 36.0, adjust delta
        # so that temp reaches exactly 36.0 instead (but do not go below).
        projected = self.temp + delta
        if projected < 36.0:
            # adjust delta to not go below 36.0
            delta = 36.0 - self.temp

        # apply update
        self.temp = float(self.temp + delta)

        # Base reward shaped to be higher when temperature is close to 37°C
        reward = -abs(self.temp - 37.0)

        info = {}
        # Side-effect penalty (small penalty to discourage risky dosing)
        if side_effect:
            reward -= 1.0
            info['side_effect'] = side_effect_mag

        # termination checks and additional penalties/bonuses
        if self.temp < 36.0 or self.temp > 41.0:
            # Danger: heavy penalty. If temp <36, add extra -5 penalty (severe hypothermia)
            self.done = True
            reward -= 10.0
            if self.temp < 36.0:
                reward -= 5.0
            info['termination'] = 'danger'
        elif 36.9 <= self.temp <= 37.1:
            # Recovered: give a bonus
            self.done = True
            reward += 10.0
            info['termination'] = 'recovered'
        else:
            self.done = False

        obs = np.array([self.temp], dtype=np.float32)
        return obs, float(reward), bool(self.done), info

    def render(self, mode='human'):
        print(f"Temperature: {self.temp:.2f}°C")

    def seed(self, seed: int | None = None):
        self._rng = np.random.RandomState(seed)
        self.seed_val = seed
