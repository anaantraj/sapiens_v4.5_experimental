#!/usr/bin/env python3
"""
Run all 4 metrics and aggregate results
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent

scripts = [
    '01_compute_jsd.py',
    '02_compute_wd.py',
    '03_compute_adaptive_recall.py',
    '04_compute_precision.py'
]

print("=" * 80)
print("RUNNING ALL METRICS")
print("=" * 80)
print("\nNOTE: Sample sizes may differ between before/after SGO due to data filtering.")
print("This is expected and metrics are computed on available data only.")
print("=" * 80)

for script in scripts:
    print(f"\n{'='*80}")
    print(f"Running: {script}")
    print(f"{'='*80}")
    
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script)],
        cwd=str(BASE_DIR)
    )
    
    if result.returncode != 0:
        print(f"\nERROR: {script} failed with exit code {result.returncode}")
        sys.exit(1)

print("\n" + "=" * 80)
print("ALL METRICS COMPLETE")
print("=" * 80)
