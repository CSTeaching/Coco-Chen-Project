"""Analyze VitalDB datasets.

Loads and explores clinical_data.csv, lab_data.csv, and track_names.csv.
"""

import pandas as pd

# Load the datasets
print("Loading datasets...")
clinical = pd.read_csv("vitaldb_data/clinical_data.csv")
lab = pd.read_csv("vitaldb_data/lab_data.csv")
tracks = pd.read_csv("vitaldb_data/track_names.csv")

print("\n" + "=" * 70)
print("CLINICAL DATA")
print("=" * 70)
print(f"\nShape: {clinical.shape}")
print(f"  - {clinical.shape[0]} cases")
print(f"  - {clinical.shape[1]} columns")

print("\nColumn names:")
for i, col in enumerate(clinical.columns, 1):
    print(f"  {i:2d}. {col}")

print("\nFirst few rows:")
print(clinical.head())

print("\n" + "=" * 70)
print("LAB DATA")
print("=" * 70)
print(f"\nShape: {lab.shape}")
print(f"  - {lab.shape[0]} lab measurements")
print(f"  - {lab.shape[1]} columns")

print("\nColumn names:")
for i, col in enumerate(lab.columns, 1):
    print(f"  {i}. {col}")

print("\nFirst few rows:")
print(lab.head())

print("\nUnique test names in lab_data:")
unique_tests = sorted(lab['name'].unique())
print(f"  Total: {len(unique_tests)} unique tests")
for i, test in enumerate(unique_tests, 1):
    count = (lab['name'] == test).sum()
    print(f"  {i:2d}. {test:20s} ({count:,} measurements)")

print("\n" + "=" * 70)
print("TRACK NAMES (Physiological Parameters)")
print("=" * 70)
print(f"\nShape: {tracks.shape}")
print(f"  - {tracks.shape[0]} available parameters")
print(f"  - {tracks.shape[1]} columns")

print("\nColumn names:")
for i, col in enumerate(tracks.columns, 1):
    print(f"  {i}. {col}")

print("\nFirst few rows:")
print(tracks.head())

print("\nFirst 20 physiological parameters and their units:")
for i in range(min(20, len(tracks))):
    param = tracks.iloc[i]['Parameter']
    desc = tracks.iloc[i]['Description']
    unit = tracks.iloc[i]['Unit']
    type_hz = tracks.iloc[i]['Type/Hz']
    print(f"  {i+1:2d}. {param:25s} | {desc:35s} | {unit:10s} | {type_hz}")

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Total patients: {clinical['subjectid'].nunique():,}")
print(f"Total cases: {clinical['caseid'].nunique():,}")
print(f"Total lab measurements: {lab.shape[0]:,}")
print(f"Available physiological parameters: {tracks.shape[0]}")
