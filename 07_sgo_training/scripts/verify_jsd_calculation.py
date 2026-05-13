#!/usr/bin/env python3
"""
Verify JSD Calculation
======================

This script verifies that JSD is being calculated correctly by:
1. Loading a sample delta file
2. Showing the actual vs predicted distributions
3. Manually calculating JSD to verify the stored values
4. Showing statistics about the distributions

Usage:
    python 07_sgo_training/scripts/verify_jsd_calculation.py --cluster cluster_1 --micro micro_0
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import entropy

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import JSD calculation function
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
import _package_setup

from calculate_behaviour_loss.weighted_topic_review_metric_calculation import calculate_jsd

EPSILON = 1e-10

def compute_jsd_manual(P: np.ndarray, Q: np.ndarray) -> float:
    """Manually compute JSD for verification."""
    # Add epsilon to avoid zeros
    P = P + EPSILON
    Q = Q + EPSILON
    
    # Re-normalize
    P = P / P.sum()
    Q = Q / Q.sum()
    
    # Compute mixture distribution
    M = 0.5 * (P + Q)
    
    # Compute KL divergences
    kl_pm = entropy(P, M, base=2)
    kl_qm = entropy(Q, M, base=2)
    
    # JSD is the average of the two KL divergences
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(jsd)


def verify_jsd_for_review(delta_entry: dict, all_themes: list = None):
    """Verify JSD calculation for a single review."""
    print("\n" + "="*80)
    print(f"Review: {delta_entry.get('review_key', 'unknown')}")
    print("="*80)
    
    predicted_themes = delta_entry.get('prediction', {}).get('predicted_themes', {})
    actual_themes = delta_entry.get('actual', {}).get('topic_probabilities', {})
    stored_jsd = delta_entry.get('theme_jsd') or delta_entry.get('theme_delta')
    
    print(f"\n📊 Stored JSD: {stored_jsd}")
    
    if not predicted_themes or not actual_themes:
        print("⚠️  Missing predicted or actual themes")
        return
    
    print(f"\n🔮 Predicted Themes Distribution:")
    pred_sum = sum(predicted_themes.values())
    print(f"   Total probability: {pred_sum:.6f}")
    for theme, prob in sorted(predicted_themes.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {theme}: {prob:.6f}")
    if len(predicted_themes) > 5:
        print(f"   ... and {len(predicted_themes) - 5} more themes")
    
    print(f"\n✅ Actual Themes Distribution (topic_probabilities):")
    act_sum = sum(actual_themes.values())
    print(f"   Total probability: {act_sum:.6f}")
    for theme, prob in sorted(actual_themes.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {theme}: {prob:.6f}")
    if len(actual_themes) > 5:
        print(f"   ... and {len(actual_themes) - 5} more themes")
    
    # Get all themes
    if all_themes is None:
        all_themes = sorted(set(list(predicted_themes.keys()) + list(actual_themes.keys())))
    
    print(f"\n📋 All Themes (n={len(all_themes)}):")
    print(f"   {', '.join(all_themes[:10])}")
    if len(all_themes) > 10:
        print(f"   ... and {len(all_themes) - 10} more")
    
    # Calculate JSD using the function
    calculated_jsd = calculate_jsd(predicted_themes, actual_themes, all_themes)
    
    # Manual calculation for verification
    P = np.array([predicted_themes.get(theme, 0.0) for theme in all_themes])
    Q = np.array([actual_themes.get(theme, 0.0) for theme in all_themes])
    
    # Normalize
    P_sum = P.sum()
    Q_sum = Q.sum()
    if P_sum > 0:
        P = P / P_sum
    if Q_sum > 0:
        Q = Q / Q_sum
    
    manual_jsd = compute_jsd_manual(P, Q)
    
    print(f"\n🧮 Calculated JSD (using function): {calculated_jsd:.10f}")
    print(f"🧮 Manual JSD (for verification):   {manual_jsd:.10f}")
    print(f"📊 Stored JSD:                       {stored_jsd:.10f}")
    
    if abs(calculated_jsd - stored_jsd) < 1e-6:
        print("✅ JSD values match!")
    else:
        print(f"⚠️  JSD mismatch! Difference: {abs(calculated_jsd - stored_jsd):.10f}")
    
    # Show distribution statistics
    print(f"\n📈 Distribution Statistics:")
    print(f"   Predicted: sum={P.sum():.6f}, min={P.min():.6f}, max={P.max():.6f}, mean={P.mean():.6f}")
    print(f"   Actual:    sum={Q.sum():.6f}, min={Q.min():.6f}, max={Q.max():.6f}, mean={Q.mean():.6f}")
    
    # Show overlap
    common_themes = set(predicted_themes.keys()) & set(actual_themes.keys())
    print(f"\n🔗 Overlap: {len(common_themes)} common themes out of {len(all_themes)} total")


def main():
    parser = argparse.ArgumentParser(description='Verify JSD calculation')
    parser.add_argument('--cluster', type=str, default='cluster_1', help='Cluster ID')
    parser.add_argument('--micro', type=str, default='micro_0', help='Micro cluster ID')
    parser.add_argument('--review-idx', type=int, default=0, help='Review index to verify (0-based)')
    parser.add_argument('--artifacts-dir', type=str, help='Path to artifacts directory')
    args = parser.parse_args()
    
    # Determine artifacts directory
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        stage_dir = Path(__file__).parent.parent
        artifacts_dir = stage_dir / "artifacts" / "sgo_train_final_predictions"
    
    # Load delta file
    delta_file = artifacts_dir / "deltas" / args.cluster / f"{args.micro}_all_reviews_deltas.json"
    
    if not delta_file.exists():
        print(f"❌ Delta file not found: {delta_file}")
        return
    
    print(f"Loading delta file: {delta_file}")
    with open(delta_file, 'r', encoding='utf-8') as f:
        delta_data = json.load(f)
    
    deltas = delta_data.get('deltas', [])
    if not deltas:
        print("❌ No deltas found in file")
        return
    
    print(f"Found {len(deltas)} reviews in delta file")
    
    # Verify the specified review
    if args.review_idx >= len(deltas):
        print(f"⚠️  Review index {args.review_idx} out of range (max: {len(deltas)-1})")
        args.review_idx = 0
    
    delta_entry = deltas[args.review_idx]
    
    # Try to get all_themes from category (would need to load from summary file)
    # For now, we'll let the function determine it from the distributions
    verify_jsd_for_review(delta_entry, all_themes=None)
    
    # Show summary statistics
    print("\n" + "="*80)
    print("Summary Statistics for All Reviews")
    print("="*80)
    
    jsd_values = []
    for entry in deltas:
        jsd = entry.get('theme_jsd') or entry.get('theme_delta')
        if jsd is not None:
            jsd_values.append(float(jsd))
    
    if jsd_values:
        print(f"\n📊 JSD Statistics (n={len(jsd_values)}):")
        print(f"   Mean:   {np.mean(jsd_values):.6f}")
        print(f"   Median: {np.median(jsd_values):.6f}")
        print(f"   Std:    {np.std(jsd_values):.6f}")
        print(f"   Min:    {np.min(jsd_values):.6f}")
        print(f"   Max:    {np.max(jsd_values):.6f}")


if __name__ == "__main__":
    main()

