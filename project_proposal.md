# Project Goal
The goal of this project is to train a reinforcement learning (RL) agent to make sequential insulin dosing decisions that maintain blood glucose levels within a clinically safe range (80–180 mg/dL) over time in a simulated diabetes patient.

Diabetes management is a sequential decision-making problem: insulin actions have delayed effects, outcomes are uncertain due to meals and metabolism, and unsafe decisions can lead to dangerous hypoglycemia or hyperglycemia. This project uses reinforcement learning to model these tradeoffs and learn treatment strategies that balance effectiveness and safety over long time horizons, using simulation parameters that are grounded through exploratory data analysis of real-life diabetes data to identify realistic glucose ranges, variability, meal effects, and insulin sensitivity trends.

# Predictive Power of the Model
The predictive power of this model lies in its ability to anticipate future blood glucose trajectories under different insulin dosing decisions and select actions that optimize long-term outcomes rather than single-step predictions.

Specifically, the trained RL agent learns to predict:
1. How blood glucose levels will evolve over time given:
    - the current glucose state
    - an insulin dose decision
    - stochastic disturbances such as meals and metabolic variability
2. The long-term risk of hypoglycemia or hyperglycemia resulting from treatment choices

