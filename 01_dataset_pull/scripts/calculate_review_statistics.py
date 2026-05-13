#!/usr/bin/env python3
"""
Script to calculate review statistics:
- Total number of reviews (train + test)
- Number of reviews in train and test sets
- Number of reviews per category
- Total number of users
- Number of users in train and test sets
"""

import json
import os
from collections import defaultdict

# File paths
TRAIN_FILE = 'train_set_reviews.json'
TEST_FILE = 'test_set_reviews.json'
FULL_FILE = 'full_user_reviews.json'

def load_json_file(filepath):
    """Load JSON file and return data."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def count_reviews_and_users(data):
    """Count reviews and users in a dataset."""
    if data is None:
        return 0, 0, defaultdict(int)
    
    total_reviews = 0
    total_users = len(data)
    category_counts = defaultdict(int)
    
    for user_id, user_data in data.items():
        reviews = user_data.get('reviews', [])
        total_reviews += len(reviews)
        
        for review in reviews:
            category = review.get('category', 'Unknown')
            if category is None:
                category = 'Unknown'
            category_counts[category] += 1
    
    return total_reviews, total_users, category_counts

def main():
    """Main execution function."""
    print("=" * 80)
    print("REVIEW STATISTICS ANALYSIS")
    print("=" * 80)
    
    # Load datasets
    train_data = load_json_file(TRAIN_FILE)
    test_data = load_json_file(TEST_FILE)
    full_data = load_json_file(FULL_FILE)
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Count train set
    train_reviews, train_users, train_categories = count_reviews_and_users(train_data)
    
    # Count test set
    test_reviews, test_users, test_categories = count_reviews_and_users(test_data)
    
    # Count full dataset (if available)
    full_reviews, full_users, full_categories = count_reviews_and_users(full_data)
    
    # Calculate totals
    total_reviews = train_reviews + test_reviews
    total_users_combined = len(set(list(train_data.keys() if train_data else []) + 
                                   list(test_data.keys() if test_data else [])))
    
    # Combine category counts
    all_categories = defaultdict(int)
    for category, count in train_categories.items():
        all_categories[category] += count
    for category, count in test_categories.items():
        all_categories[category] += count
    
    # Print results
    print(f"\n TOTAL REVIEWS:")
    print(f"   Train set: {train_reviews:,}")
    print(f"   Test set:  {test_reviews:,}")
    print(f"   Combined:  {total_reviews:,}")
    if full_data:
        print(f"   Full file: {full_reviews:,}")
    
    print(f"\n TOTAL USERS:")
    print(f"   Train set: {train_users:,}")
    print(f"   Test set:  {test_users:,}")
    print(f"   Combined (unique): {total_users_combined:,}")
    if full_data:
        print(f"   Full file: {full_users:,}")
    
    print(f"\n REVIEWS BY CATEGORY:")
    print(f"   {'Category':<40} {'Train':<12} {'Test':<12} {'Total':<12}")
    print(f"   {'-'*40} {'-'*12} {'-'*12} {'-'*12}")
    
    # Sort categories by total count
    sorted_categories = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)
    
    for category, total_count in sorted_categories:
        train_count = train_categories.get(category, 0)
        test_count = test_categories.get(category, 0)
        category_str = str(category) if category is not None else 'Unknown'
        print(f"   {category_str:<40} {train_count:<12,} {test_count:<12,} {total_count:<12,}")
    
    print(f"\n   {'TOTAL':<40} {train_reviews:<12,} {test_reviews:<12,} {total_reviews:<12,}")
    
    # Save results to file
    results = {
        "total_reviews": {
            "train": train_reviews,
            "test": test_reviews,
            "combined": total_reviews,
            "full_file": full_reviews if full_data else None
        },
        "total_users": {
            "train": train_users,
            "test": test_users,
            "combined": total_users_combined,
            "full_file": full_users if full_data else None
        },
        "reviews_by_category": {
            category: {
                "train": train_categories.get(category, 0),
                "test": test_categories.get(category, 0),
                "total": count
            }
            for category, count in sorted_categories
        }
    }
    
    output_file = 'review_statistics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n Results saved to: {output_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()

