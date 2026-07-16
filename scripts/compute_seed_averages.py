#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

ARTIFACTS = Path('artifacts')
AGG_CSV = ARTIFACTS / 'final_dqn_aggregated.csv'
PER_SEED_CSV = ARTIFACTS / 'seed_metrics_per_eval_run.csv'
SEED_AVG_CSV = ARTIFACTS / 'seed_averaged_metrics.csv'
SEED_AVG_MD = ARTIFACTS / 'seed_averaged_metrics.md'

if not AGG_CSV.exists():
    raise SystemExit(f"Missing {AGG_CSV}")

df = pd.read_csv(AGG_CSV)
# consider only evaluation rows
eval_df = df[df['phase'].str.lower() == 'eval'].copy()
if eval_df.empty:
    raise SystemExit('No eval rows found in aggregated CSV')

# metrics of interest
metrics = ['time_in_range_percent', 'hypo_events', 'total_insulin_units']

# Per-seed (source) stats across eval runs
per_seed = eval_df.groupby('source')[metrics].agg(['mean', 'std', 'count']).reset_index()
# flatten columns
per_seed.columns = ['_'.join(filter(None, col)).rstrip('_') for col in per_seed.columns.values]
per_seed = per_seed.rename(columns={'source_': 'source'})
per_seed.to_csv(PER_SEED_CSV, index=False)

# Now compute seed-averaged metrics: for each metric, take mean and std across seeds using the per-seed means
seed_means = per_seed[[c for c in per_seed.columns if c.endswith('mean')]].copy()
# map column names back to metric names
metric_names = [c.replace('_mean','') for c in seed_means.columns if c.endswith('_mean')]

rows = []
for m in metric_names:
    col = f'{m}_mean'
    vals = seed_means[col].dropna().values
    rows.append({'metric': m,
                 'mean_across_seeds': float(np.mean(vals)) if len(vals)>0 else float('nan'),
                 'std_across_seeds': float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0,
                 'units': 'percent' if 'time_in_range' in m else ('events' if 'hypo' in m else 'units')})

seed_avg_df = pd.DataFrame(rows)
seed_avg_df.to_csv(SEED_AVG_CSV, index=False)

# Also write a Markdown summary
with open(SEED_AVG_MD, 'w') as f:
    f.write('# Seed-averaged Final Metrics\n\n')
    f.write('Mean ± std across seeds (computed from per-seed eval means)\n\n')
    f.write('| Metric | Mean | Std | Units |\n')
    f.write('|---|---:|---:|---|\n')
    for _, r in seed_avg_df.iterrows():
        f.write(f"| {r.metric} | {r.mean_across_seeds:.3f} | {r.std_across_seeds:.3f} | {r.units} |\n")

print('Wrote:', PER_SEED_CSV, SEED_AVG_CSV, SEED_AVG_MD)
