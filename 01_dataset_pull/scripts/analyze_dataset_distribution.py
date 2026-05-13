#!/usr/bin/env python3
"""
Analyze Dataset Distribution
=============================

Analyzes the sampled dataset and shows:
- Number of users in each category
- Number of reviews in each category
- Common users across categories
- Category overlap statistics
- User review distribution

Usage:
    python 01_dataset_pull/scripts/analyze_dataset_distribution.py
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.wandb_utils import get_stage_config, get_artifact_dir


def map_category_to_main_category(category_name: str) -> str:
    """Map category to one of the 7 main categories."""
    if not category_name:
        return None
    
    category_lower = category_name.lower()
    
    # Mapping rules
    if any(x in category_lower for x in ['fashion', 'clothing', 'shoes', 'jewelry', 'apparel', 'amazon fashion']):
        return "Clothing_Shoes_and_Jewelry"
    elif any(x in category_lower for x in ['appliance', 'tools & home improvement', 'home improvement', 
                                          'amazon home', 'industrial & scientific']):
        return "Appliances"
    elif any(x in category_lower for x in ['beauty', 'cosmetic', 'makeup', 'skincare', 'premium beauty', 
                                          'all beauty']):
        return "All_Beauty"
    elif any(x in category_lower for x in ['digital music', 'music', 'musical instruments', 
                                          'portable audio', 'home audio', 'car electronics']):
        return "Digital_Music"
    elif any(x in category_lower for x in ['video game', 'video games', 'gaming', 'game', 'games', 
                                          'toys & games', 'toys and games']):
        return "Video_Games"
    elif any(x in category_lower for x in ['health', 'personal care', 'wellness', 'health & personal care',
                                          'baby', 'grocery']):
        return "Health_and_Personal_Care"
    elif any(x in category_lower for x in ['software', 'appstore', 'app store', 'application', 'applications',
                                          'computers', 'all electronics', 'cell phones', 'camera', 'photo',
                                          'office products', 'books', 'kindle', 'movies', 'tv']):
        return "Software"
    elif any(x in category_lower for x in ['sports', 'outdoors', 'automotive', 'arts', 'crafts', 'sewing']):
        return "Appliances"
    
    return None


def analyze_dataset(file_path: Path, output_json: Path = None):
    """Analyze the dataset and print statistics."""
    
    print("=" * 80)
    print("DATASET DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_users = len(data)
    print(f"Total users: {total_users}")
    
    # Dictionary to store all statistics for JSON output
    stats = {
        "dataset_info": {},
        "category_statistics": {},
        "original_category_distribution": {},
        "user_overlap": {},
        "reviews_per_user_distribution": {},
        "summary_statistics": {}
    }
    
    # Collect statistics
    category_reviews = defaultdict(int)  # category -> review count
    category_users = defaultdict(set)   # category -> set of user_ids
    user_categories = defaultdict(set)  # user_id -> set of categories
    user_review_counts = defaultdict(int)  # user_id -> total review count
    original_category_reviews = defaultdict(int)  # original category name -> review count
    
    for user_id, user_data in data.items():
        reviews = user_data.get('reviews', [])
        user_review_counts[user_id] = len(reviews)
        
        for review in reviews:
            original_cat = review.get('category', 'Unknown')
            if not original_cat or original_cat == 'None':
                original_cat = 'Unknown'
            
            original_category_reviews[original_cat] += 1
            
            # Map to main category
            main_cat = map_category_to_main_category(original_cat)
            if main_cat:
                category_reviews[main_cat] += 1
                category_users[main_cat].add(user_id)
                user_categories[user_id].add(main_cat)
    
    total_reviews = sum(category_reviews.values())
    
    # Store dataset info
    stats["dataset_info"] = {
        "total_users": total_users,
        "total_reviews": total_reviews,
        "source_file": str(file_path)
    }
    
    # Print category statistics
    print(f"\nTotal reviews: {total_reviews}")
    print("\n" + "=" * 80)
    print("CATEGORY STATISTICS (7 Main Categories)")
    print("=" * 80)
    
    print(f"\n{'Category':<30} {'Users':<12} {'Reviews':<12} {'Avg Reviews/User':<15} {'% of Total':<12}")
    print("-" * 80)
    
    category_stats_list = []
    for category in sorted(category_reviews.keys()):
        users_count = len(category_users[category])
        reviews_count = category_reviews[category]
        avg_reviews = reviews_count / users_count if users_count > 0 else 0
        percentage = (reviews_count / total_reviews * 100) if total_reviews > 0 else 0
        
        category_stat = {
            "category": category,
            "users": users_count,
            "reviews": reviews_count,
            "avg_reviews_per_user": round(avg_reviews, 2),
            "percentage_of_total": round(percentage, 2)
        }
        category_stats_list.append(category_stat)
        
        print(f"{category:<30} {users_count:<12} {reviews_count:<12} {avg_reviews:<15.2f} {percentage:<12.2f}%")
    
    stats["category_statistics"] = category_stats_list
    
    # Original category distribution
    print("\n" + "=" * 80)
    print("ORIGINAL CATEGORY DISTRIBUTION (All Categories)")
    print("=" * 80)
    print(f"\n{'Category':<40} {'Reviews':<12} {'% of Total':<12}")
    print("-" * 80)
    
    original_cat_stats = []
    for category, count in sorted(original_category_reviews.items(), key=lambda x: -x[1]):
        percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
        original_cat_stats.append({
            "category": category,
            "reviews": count,
            "percentage_of_total": round(percentage, 2)
        })
        print(f"{category:<40} {count:<12} {percentage:<12.2f}%")
    
    stats["original_category_distribution"] = original_cat_stats
    
    # User overlap analysis
    print("\n" + "=" * 80)
    print("USER OVERLAP ANALYSIS")
    print("=" * 80)
    
    # Count users by number of categories they appear in
    users_by_category_count = Counter(len(cats) for cats in user_categories.values())
    
    print(f"\nUsers by number of categories:")
    print(f"{'# Categories':<15} {'# Users':<12} {'% of Users':<12}")
    print("-" * 40)
    
    users_by_cat_count_list = []
    for num_cats in sorted(users_by_category_count.keys()):
        user_count = users_by_category_count[num_cats]
        percentage = (user_count / total_users * 100) if total_users > 0 else 0
        users_by_cat_count_list.append({
            "num_categories": num_cats,
            "num_users": user_count,
            "percentage_of_users": round(percentage, 2)
        })
        print(f"{num_cats:<15} {user_count:<12} {percentage:<12.2f}%")
    
    # Category pairs overlap
    print(f"\nCategory Pair Overlaps (Top 10):")
    print(f"{'Category 1':<30} {'Category 2':<30} {'Common Users':<15}")
    print("-" * 80)
    
    categories_list = sorted(category_users.keys())
    overlaps = []
    overlap_list = []
    for i, cat1 in enumerate(categories_list):
        for cat2 in categories_list[i+1:]:
            common = len(category_users[cat1] & category_users[cat2])
            if common > 0:
                overlaps.append((cat1, cat2, common))
                overlap_list.append({
                    "category_1": cat1,
                    "category_2": cat2,
                    "common_users": common
                })
    
    overlaps.sort(key=lambda x: -x[2])
    for cat1, cat2, common in overlaps[:10]:
        print(f"{cat1:<30} {cat2:<30} {common:<15}")
    
    stats["user_overlap"] = {
        "users_by_category_count": users_by_cat_count_list,
        "category_pair_overlaps": sorted(overlap_list, key=lambda x: -x["common_users"])
    }
    
    # Reviews per user distribution
    print("\n" + "=" * 80)
    print("REVIEWS PER USER DISTRIBUTION")
    print("=" * 80)
    
    review_count_distribution = Counter(user_review_counts.values())
    
    print(f"\n{'# Reviews':<15} {'# Users':<12} {'% of Users':<12}")
    print("-" * 40)
    
    reviews_per_user_list = []
    for num_reviews in sorted(review_count_distribution.keys()):
        user_count = review_count_distribution[num_reviews]
        percentage = (user_count / total_users * 100) if total_users > 0 else 0
        reviews_per_user_list.append({
            "num_reviews": num_reviews,
            "num_users": user_count,
            "percentage_of_users": round(percentage, 2)
        })
        print(f"{num_reviews:<15} {user_count:<12} {percentage:<12.2f}%")
    
    stats["reviews_per_user_distribution"] = reviews_per_user_list
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    avg_reviews_per_user = total_reviews / total_users if total_users > 0 else 0
    max_reviews_per_user = max(user_review_counts.values()) if user_review_counts else 0
    min_reviews_per_user = min(user_review_counts.values()) if user_review_counts else 0
    
    users_in_multiple_categories = sum(1 for cats in user_categories.values() if len(cats) > 1)
    users_in_single_category = total_users - users_in_multiple_categories
    
    print(f"\nTotal Users: {total_users}")
    print(f"Total Reviews: {total_reviews}")
    print(f"Average Reviews per User: {avg_reviews_per_user:.2f}")
    print(f"Min Reviews per User: {min_reviews_per_user}")
    print(f"Max Reviews per User: {max_reviews_per_user}")
    print(f"\nUsers in Single Category: {users_in_single_category} ({users_in_single_category/total_users*100:.2f}%)")
    print(f"Users in Multiple Categories: {users_in_multiple_categories} ({users_in_multiple_categories/total_users*100:.2f}%)")
    print(f"Number of Main Categories: {len(category_reviews)}")
    print(f"Number of Original Categories: {len(original_category_reviews)}")
    
    stats["summary_statistics"] = {
        "total_users": total_users,
        "total_reviews": total_reviews,
        "average_reviews_per_user": round(avg_reviews_per_user, 2),
        "min_reviews_per_user": min_reviews_per_user,
        "max_reviews_per_user": max_reviews_per_user,
        "users_in_single_category": users_in_single_category,
        "users_in_single_category_percentage": round(users_in_single_category/total_users*100, 2),
        "users_in_multiple_categories": users_in_multiple_categories,
        "users_in_multiple_categories_percentage": round(users_in_multiple_categories/total_users*100, 2),
        "number_of_main_categories": len(category_reviews),
        "number_of_original_categories": len(original_category_reviews)
    }
    
    # Save to JSON file
    if output_json is None:
        output_json = file_path.parent / "dataset_distribution_stats.json"
    
    print("\n" + "=" * 80)
    print(f"Saving statistics to: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[OK] Statistics saved to {output_json}")
    print("=" * 80)


def main():
    """Main execution function."""
    # Get config to find artifact path
    cfg = get_stage_config("01_dataset_pull")
    output_artifact_name = cfg.get("output_artifact", "amazon_reviews_v1")
    
    artifact_dir = get_artifact_dir("01_dataset_pull", output_artifact_name)
    file_path = artifact_dir / "full_user_reviews.json"
    output_json = artifact_dir / "dataset_distribution_stats.json"
    
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        print("Please run the dataset pull script first.")
        return
    
    analyze_dataset(file_path, output_json)


if __name__ == "__main__":
    main()

