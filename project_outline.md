# Final Project Goal
By the end of April, this project will produce a reinforcement learning agent that can manage blood glucose levels in a simulated diabetes patient. The simulation will be grounded using empirical trends extracted from real-life diabetes data, and the learned policy will be evaluated using clinically meaningful metrics such as time-in-range and safety violations.

# BI-WEEKLY PHASE 1: PROJECT SETUP AND DATA SELECTION
Define project scope and research goal
Decide state, action, and reward at high level
Select real-life diabetes datasets for analysis
Set up clean project directory structure

https://www.kaggle.com/datasets/ryanmouton/ohiot1dm

# BI-WEEKLY PHASE 2: REAL-LIFE DATA ANALYSIS
Perform exploratory data analysis on diabetes data
Identify realistic glucose ranges
Measure glucose variability and fluctuations
Estimate meal-related glucose effects
Estimate insulin sensitivity trends
Document extracted trends for parameter grounding

# BI-WEEKLY PHASE 3: DATA-GROUNDED SIMULATION DESIGN
Design diabetes simulation environment
Set simulation parameters based on data analysis results
Implement glucose dynamics, insulin effects, and meals
Define safety thresholds for hypo- and hyperglycemia
Validate simulation behavior with test runs

# BI-WEEKLY PHASE 4: REINFORCEMENT LEARNING IMPLEMENTATION
Implement tabular reinforcement learning agent
Define reward function emphasizing safety and stability
Train agent in grounded simulation environment
Verify learning behavior and convergence

# BI-WEEKLY PHASE 5: BASELINES AND EVALUATION
Implement data-informed rule-based baselines
Compare RL agent to baseline policies
Evaluate time-in-range performance
Evaluate hypoglycemia and hyperglycemia events

# BI-WEEKLY PHASE 6: VALIDATION AND SENSITIVITY ANALYSIS
Test robustness to parameter changes
Analyze performance across different variability settings
Perform optional distribution matching with real-life data

# BI-WEEKLY PHASE 7: FINALIZATION
Finalize experiments and results
Select final plots and metrics
Clean and document codebase