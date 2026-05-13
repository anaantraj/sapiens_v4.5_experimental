#!/usr/bin/env python3
"""
Task 3: Show metrics by category of product cut
Analyzes all metrics (JSD, Precision, Recall, WD) grouped by product category
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import yaml
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust BASE_DIR since script is in a subfolder
BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Load config
with open(BASE_DIR / '13_metrics_analysis/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = BASE_DIR / config['output_dir']
ARTIFACTS_DIR = OUTPUT_DIR
GRAPH_OUTPUT_DIR = BASE_DIR / '13_metrics_analysis'

# Filter: Only analyze these categories
SELECTED_CATEGORIES = {
    'All Beauty',
    'Appliances',
    'AMAZON FASHION',  # Clothing, Shoes and Jewellery
    'Digital Music',
    'Health & Personal Care',
    'Software',
    'Video Games'
}


def validate_file(file_path: Path, file_name: str) -> bool:
    """Validate file exists and is readable."""
    if not file_path.exists():
        print(f"  ERROR: {file_name} not found at {file_path}")
        return False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"  ERROR: {file_name} is not valid JSON: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: Cannot read {file_name}: {e}")
        return False


def normalize_category(category: str) -> str:
    """Normalize category name (keep as-is per user request)."""
    if not category:
        return 'unknown'
    return category.strip()


def load_metric_file(file_path: Path, metric_name: str) -> List[Dict]:
    """Load and validate metric file."""
    print(f"  Loading {metric_name}...")
    if not validate_file(file_path, metric_name):
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"  ERROR: {metric_name} should be a list, got {type(data)}")
            return []
        
        # Validate structure
        valid_records = []
        invalid_count = 0
        missing_category = 0
        
        for record in data:
            if not isinstance(record, dict):
                invalid_count += 1
                continue
            
            # Check required fields
            if 'user_id' not in record:
                invalid_count += 1
                continue
            
            # Check category
            category = record.get('category')
            if not category:
                missing_category += 1
                record['category'] = 'unknown'
            else:
                record['category'] = normalize_category(category)
            
            # Check metric value exists
            metric_fields = ['jsd', 'precision_at_k', 'adaptive_recall_at_k', 'wd']
            has_metric = any(field in record for field in metric_fields)
            if not has_metric:
                invalid_count += 1
                continue
            
            valid_records.append(record)
        
        if invalid_count > 0:
            print(f"  WARNING: {invalid_count} invalid records skipped")
        if missing_category > 0:
            print(f"  WARNING: {missing_category} records with missing category (marked as 'unknown')")
        
        print(f"  Loaded {len(valid_records)} valid records")
        return valid_records
        
    except Exception as e:
        print(f"  ERROR: Failed to load {metric_name}: {e}")
        return []


def extract_metric_value(record: Dict, metric_name: str) -> Optional[float]:
    """Extract metric value from record, handling different field names."""
    field_mapping = {
        'jsd': ['jsd'],
        'precision': ['precision_at_k'],
        'recall': ['adaptive_recall_at_k'],
        'wd': ['wd', 'wasserstein_distance']
    }
    
    fields = field_mapping.get(metric_name, [metric_name])
    
    for field in fields:
        if field in record:
            value = record[field]
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    return None
                return float(value)
    
    return None


def analyze_by_category(metric_data: List[Dict], metric_name: str) -> Dict:
    """Analyze metrics grouped by category."""
    print(f"\nAnalyzing {metric_name} by category...")
    
    # Group by category
    category_metrics = defaultdict(list)
    no_metric_count = 0
    
    for record in metric_data:
        category = record.get('category', 'unknown')
        category = normalize_category(category)
        
        # Filter: Only include selected categories
        if category not in SELECTED_CATEGORIES:
            continue
        
        # Get metric value
        metric_value = extract_metric_value(record, metric_name)
        if metric_value is None:
            no_metric_count += 1
            continue
        
        category_metrics[category].append(metric_value)
    
    if no_metric_count > 0:
        print(f"  WARNING: {no_metric_count} records without valid metric value")
    
    # Compute statistics per category
    results = {}
    for category, values in sorted(category_metrics.items()):
        if not values:
            continue
        
        values_array = np.array(values)
        results[category] = {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'median': float(np.median(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'q25': float(np.percentile(values_array, 25)),
            'q75': float(np.percentile(values_array, 75))
        }
    
    print(f"  Analyzed {len(results)} categories")
    return results


def create_visualizations(all_results: Dict):
    """Create visualizations for metrics by category."""
    print("\nCreating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    metrics_display = {
        'jsd': 'JSD',
        'precision': 'Precision@K',
        'recall': 'Adaptive Recall@K',
        'wd': 'Wasserstein Distance'
    }
    
    for metric_name, results in all_results.items():
        if not results:
            continue
        
        # Prepare data
        categories = []
        means = []
        stds = []
        
        sorted_categories = sorted(results.items(), key=lambda x: x[1]['mean'])
        for category, stats in sorted_categories:
            categories.append(category)
            means.append(stats['mean'])
            stds.append(stats['std'])
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                     color='steelblue', edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metrics_display.get(metric_name, metric_name)}', fontsize=12, fontweight='bold')
        ax.set_title(f'{metrics_display.get(metric_name, metric_name)} by Category', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        output_file = GRAPH_OUTPUT_DIR / f'metrics_by_category_{metric_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()


def main():
    """Main execution."""
    print("=" * 80)
    print("TASK 3: METRICS BY CATEGORY")
    print("=" * 80)
    
    # Load all metric files (after SGO)
    metrics = {
        'jsd': load_metric_file(ARTIFACTS_DIR / 'after_sgo_jsd.json', 'JSD'),
        'precision': load_metric_file(ARTIFACTS_DIR / 'after_sgo_precision.json', 'Precision'),
        'recall': load_metric_file(ARTIFACTS_DIR / 'after_sgo_adaptive_recall.json', 'Recall'),
        'wd': load_metric_file(ARTIFACTS_DIR / 'after_sgo_wd.json', 'Wasserstein Distance')
    }
    
    # Analyze each metric by category
    all_results = {}
    for metric_name, metric_data in metrics.items():
        if not metric_data:
            print(f"\nWARNING: Skipping {metric_name} - no data")
            continue
        
        results = analyze_by_category(metric_data, metric_name)
        all_results[metric_name] = results
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS: METRICS BY CATEGORY")
    print("=" * 80)
    
    for metric_name, results in all_results.items():
        print(f"\n{metric_name.upper()} by Category:")
        print("-" * 80)
        print(f"{'Category':<30} {'Count':<10} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 80)
        
        # Sort by mean value
        sorted_categories = sorted(results.items(), key=lambda x: x[1]['mean'])
        
        for category, stats in sorted_categories:
            print(f"{category:<30} {stats['count']:<10} {stats['mean']:<12.4f} {stats['median']:<12.4f} "
                  f"{stats['std']:<12.4f} {stats['min']:<12.4f} {stats['max']:<12.4f}")
    
    # Create visualizations
    create_visualizations(all_results)
    
    # Save results
    output_file = ARTIFACTS_DIR / 'metrics_by_category.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
