# Category-Specific Theme Universe Issue

## The Problem

**Themes belong to different categories**, and each category has its own theme universe:

- **Fashion Category**: Themes like "Fit & Sizing Accuracy", "Material & Fabric Performance", "Style & Appearance"
- **Video Games Category**: Themes like "Gameplay", "Graphics", "Story", "Performance"  
- **Beauty Category**: Themes like "Product Quality", "Packaging", "Skin Compatibility"

### Current Issue in Our Code

**Our Approach 2 currently aggregates ALL themes together:**
```python
# We're doing this (WRONG for cross-category):
aggregated_actual = defaultdict(float)  # All themes mixed together
aggregated_predicted = defaultdict(float)  # All themes mixed together

for review in all_reviews:  # Mixing Fashion + Games + Beauty reviews
    for theme, prob in actual_themes.items():
        aggregated_actual[theme] += prob  # "Fit & Sizing" + "Gameplay" + "Product Quality" all together!
```

**This is problematic because:**
1. We're mixing themes from different categories
2. Fashion themes and Game themes are not comparable
3. The resulting distribution doesn't make semantic sense
4. JSD calculation becomes meaningless when comparing incompatible theme spaces

---

## What Should Be Done

### Option 1: Aggregate by Category, Then Calculate JSD Per Category (RECOMMENDED)

**For Approach 2, we should:**

1. **Group reviews by category**
2. **For each category separately:**
   - Aggregate actual distributions (only themes for that category)
   - Aggregate predicted distributions (only themes for that category)
   - Calculate JSD for that category
3. **Report JSD per category**

**Example:**
```python
# Per category aggregation
category_aggregated = {
    "AMAZON FASHION": {
        'actual': {"Fit & Sizing": 0.4, "Material": 0.3, "Style": 0.3},
        'predicted': {"Fit & Sizing": 0.35, "Material": 0.35, "Style": 0.3},
        'jsd': 0.012
    },
    "Video Games": {
        'actual': {"Gameplay": 0.5, "Graphics": 0.3, "Story": 0.2},
        'predicted': {"Gameplay": 0.45, "Graphics": 0.35, "Story": 0.2},
        'jsd': 0.008
    }
}
```

**Pros:**
- Mathematically correct (comparing same theme universe)
- Semantically meaningful
- Shows which categories have better/worse predictions
- Can aggregate category-level JSDs if needed (weighted average)

**Cons:**
- More complex (need to track by category)
- Multiple JSD values (one per category)

---

### Option 2: Only Aggregate Reviews from Same Category (Current Approach 1 is OK)

**For Approach 1 (Per-Review JSD):**
- This is **already correct** because:
  - Each review has its own category
  - We calculate JSD per review (within same category)
  - Then aggregate the JSD values (which is fine - JSD values are comparable)

**For Approach 2:**
- We could **only aggregate reviews from the same category**
- But this would require filtering or grouping first
- Still need to decide: one JSD per category, or weighted average?

---

### Option 3: Use Category-Agnostic Aggregation (Current - NOT RECOMMENDED)

**What we're currently doing:**
- Mix all themes from all categories
- Calculate one global JSD

**Why this is problematic:**
- Themes from different categories are not comparable
- "Fit & Sizing" (Fashion) and "Gameplay" (Games) are completely different concepts
- The aggregate distribution doesn't represent anything meaningful
- JSD value becomes hard to interpret

**When it might be acceptable:**
- If you want a "rough overall similarity" metric
- If categories have overlapping themes (unlikely in your case)
- For exploratory analysis only

---

## Recommended Solution

### For Approach 2: Category-Level Aggregation

**Structure:**
```python
{
    'by_category': {
        'AMAZON FASHION': {
            'jsd': 0.012,
            'actual_distribution': {...},
            'predicted_distribution': {...},
            'mixture_distribution': {...},
            'review_count': 856
        },
        'Video Games': {
            'jsd': 0.008,
            'actual_distribution': {...},
            'predicted_distribution': {...},
            'mixture_distribution': {...},
            'review_count': 423
        },
        ...
    },
    'weighted_average_jsd': 0.010,  # Optional: weighted by review count
    'overall_jsd': None  # Cannot compute (different theme universes)
}
```

**Visualization:**
- One graph per category showing distributions
- Bar chart comparing JSD across categories
- Summary table with JSD per category

---

## Comparison with Their Method

**Their code also has this issue** - they're aggregating across all categories too:
```python
def build_distribution(reviews, source_type):
    # They also mix all themes together
    for r in reviews:  # All categories mixed
        themes = r.get('theme_token_probabilities', {})
        for t, p in themes.items():
            sums[t] += float(p)  # Mixing Fashion + Games + Beauty themes
```

**So their method has the same problem!**

---

## Summary

**Current State:**
- ✅ Approach 1 (Per-Review JSD): **Correct** - each review is within its category
- ❌ Approach 2 (Aggregate JSD): **Problematic** - mixing themes from different categories

**What Should Be Done:**
1. **Keep Approach 1 as-is** (it's correct)
2. **Modify Approach 2** to aggregate by category first
3. **Calculate JSD per category** (one JSD value per category)
4. **Report category-level results** with visualizations per category
5. **Optionally compute weighted average JSD** across categories

**Key Insight:**
- Themes are category-specific
- Cannot meaningfully aggregate across categories
- Must compute JSD within each category's theme universe
- Then can compare/aggregate the JSD values (not the distributions)


















