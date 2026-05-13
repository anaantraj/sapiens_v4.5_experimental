#!/usr/bin/env python3
"""
Combined script to build delta convergence matrices and create visualizations.

This script:
1. Builds delta matrices from I0 to I12
2. Creates convergence visualizations
3. Saves all outputs to delta_convergence_analysis folder
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

def main():
    """Run both matrix building and visualization"""
    print("="*70)
    print("DELTA CONVERGENCE ANALYSIS - FULL PIPELINE")
    print("="*70)
    
    # Step 1: Build matrices
    print("\n" + "="*70)
    print("STEP 1: Building Delta Matrices")
    print("="*70)
    try:
        from build_delta_convergence_matrix import main as build_main
        build_main()
    except Exception as e:
        print(f"Error building matrices: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Create visualizations
    print("\n" + "="*70)
    print("STEP 2: Creating Visualizations")
    print("="*70)
    try:
        from visualize_delta_convergence import main as viz_main
        viz_main()
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Matrices were built successfully. You can run visualize_delta_convergence.py separately.")
        return
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("All outputs saved to: delta_convergence_analysis/")
    print("  - CSV files: delta_matrix_*.csv")
    print("  - Excel file: delta_matrices_all.xlsx")
    print("  - Metrics: convergence_metrics.csv")
    print("  - Figures: figures/*.png")

if __name__ == "__main__":
    main()








