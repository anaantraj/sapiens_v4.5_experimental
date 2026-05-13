# JSD Calculation Methods: Detailed Explanation

## Overview

There are **3 different methods** being used to calculate JSD. Each measures something slightly different.

---

## METHOD 1: Our Approach 1 (Per-Review JSD, Then Aggregate)

### What It Does:
1. **For each review individually:**
   - Extract `actual.predicted_themes` → Distribution P (e.g., {"Theme1": 0.4, "Theme2": 0.6})
   - Extract `prediction.predicted_themes` → Distribution Q (e.g., {"Theme1": 0.3, "Theme2": 0.7})
   - Calculate JSD(P, Q) for that review → One JSD value per review

2. **Then aggregate the JSD values:**
   - Mean JSD = average of all per-review JSD values
   - Median, std, etc.

### Example:
```
Review 1: Actual = {A:0.8, B:0.2}, Predicted = {A:0.6, B:0.4} → JSD = 0.15
Review 2: Actual = {A:0.3, B:0.7}, Predicted = {A:0.5, B:0.5} → JSD = 0.12
Review 3: Actual = {A:0.9, B:0.1}, Predicted = {A:0.7, B:0.3} → JSD = 0.18

Result: Mean JSD = (0.15 + 0.12 + 0.18) / 3 = 0.15
```

### What It Measures:
- **Average divergence per review**
- "On average, how different are the distributions for each individual review?"

### Pros:
- Captures per-review variation
- Shows distribution of JSD values across reviews
- Good for understanding individual review performance

### Cons:
- Doesn't show overall distribution similarity
- Can be misleading if some reviews are very different but others are very similar

---

## METHOD 2: Our Approach 2 (Aggregate Distributions, Then JSD)

### What It Does:
1. **Aggregate all actual distributions:**
   - Sum probabilities for each theme across all reviews
   - Normalize to get one aggregate actual distribution

2. **Aggregate all predicted distributions:**
   - Sum probabilities for each theme across all reviews
   - Normalize to get one aggregate predicted distribution

3. **Calculate JSD once:**
   - JSD(aggregate_actual, aggregate_predicted) → One JSD value total

### Example:
```
Review 1: Actual = {A:0.8, B:0.2}, Predicted = {A:0.6, B:0.4}
Review 2: Actual = {A:0.3, B:0.7}, Predicted = {A:0.5, B:0.5}
Review 3: Actual = {A:0.9, B:0.1}, Predicted = {A:0.7, B:0.3}

Aggregate Actual:
  A: (0.8 + 0.3 + 0.9) / 3 = 0.667
  B: (0.2 + 0.7 + 0.1) / 3 = 0.333
  Normalize: {A: 0.667, B: 0.333} (already normalized)

Aggregate Predicted:
  A: (0.6 + 0.5 + 0.7) / 3 = 0.6
  B: (0.4 + 0.5 + 0.3) / 3 = 0.4
  Normalize: {A: 0.6, B: 0.4} (already normalized)

Result: JSD({A:0.667, B:0.333}, {A:0.6, B:0.4}) = 0.008
```

### What It Measures:
- **Overall distribution similarity**
- "How similar are the aggregate distributions of all reviews?"

### Pros:
- Shows overall distribution match
- Single number that represents total divergence
- Good for comparing different methods/models

### Cons:
- Loses per-review granularity
- Can hide individual review issues

### How We Aggregate:
```python
# We SUM probabilities (since each review is already a probability distribution)
for review in reviews:
    for theme, prob in actual_themes.items():
        aggregated_actual[theme] += prob  # SUM

# Then normalize once
total = sum(aggregated_actual.values())
aggregated_actual = {theme: prob/total for theme, prob in aggregated_actual.items()}
```

**Why summing works:**
- Each review's distribution sums to 1.0
- Summing preserves the total probability mass
- Normalizing once gives the correct aggregate distribution

---

## METHOD 3: Their Method (Average Probabilities, Then JSD)

### What It Does:
1. **For each theme, average its probability across reviews:**
   - Count how many reviews have each theme
   - Average the probabilities for that theme
   - This gives average probability per theme

2. **Build distributions from averages:**
   - Aggregate actual = {theme: average_probability across reviews}
   - Aggregate predicted = {theme: average_probability across reviews}

3. **Calculate JSD once:**
   - JSD(aggregate_actual, aggregate_predicted)

### Example:
```
Review 1: Actual = {A:0.8, B:0.2}, Predicted = {A:0.6, B:0.4}
Review 2: Actual = {A:0.3, B:0.7}, Predicted = {A:0.5, B:0.5}
Review 3: Actual = {A:0.9, B:0.1}, Predicted = {A:0.7, B:0.3}

Aggregate Actual (AVERAGING):
  A: (0.8 + 0.3 + 0.9) / 3 = 0.667  ← Average of A probabilities
  B: (0.2 + 0.7 + 0.1) / 3 = 0.333  ← Average of B probabilities
  Already normalized: {A: 0.667, B: 0.333}

Aggregate Predicted (AVERAGING):
  A: (0.6 + 0.5 + 0.7) / 3 = 0.6
  B: (0.4 + 0.5 + 0.3) / 3 = 0.4
  Already normalized: {A: 0.6, B: 0.4}

Result: JSD({A:0.667, B:0.333}, {A:0.6, B:0.4}) = 0.008
```

### What It Measures:
- **Average probability distribution**
- "What's the average probability assigned to each theme?"

### How They Aggregate:
```python
sums = defaultdict(float)
cnts = defaultdict(int)  # Count how many reviews have each theme

for review in reviews:
    for theme, prob in themes.items():
        sums[theme] += prob
        cnts[theme] += 1  # Count occurrences

# AVERAGE probabilities
dist = {theme: sums[theme]/cnts[theme] if cnts[theme] > 0 else 0.0 
        for theme in all_themes}

# Then normalize
total = sum(dist.values())
dist = {theme: prob/total for theme, prob in dist.items()}
```

---

## KEY DISTINCTION: Summing vs Averaging

### When They're The Same:
If **every review has every theme** (all themes appear in all reviews), then:
- **Summing then normalizing** = **Averaging then normalizing**
- Both give the same result

### When They're Different:
If **some themes appear in fewer reviews**, then:

**Example:**
```
Review 1: {A:0.8, B:0.2, C:0.0}  (C not present)
Review 2: {A:0.3, B:0.7, C:0.0}  (C not present)
Review 3: {A:0.5, B:0.3, C:0.2}  (C present)

Our Method (SUMMING):
  A: 0.8 + 0.3 + 0.5 = 1.6 → normalize → 0.64
  B: 0.2 + 0.7 + 0.3 = 1.2 → normalize → 0.48
  C: 0.0 + 0.0 + 0.2 = 0.2 → normalize → 0.08
  Total: 3.0 → normalized to 1.0

Their Method (AVERAGING):
  A: (0.8 + 0.3 + 0.5) / 3 = 0.533 → normalize → 0.64
  B: (0.2 + 0.7 + 0.3) / 3 = 0.400 → normalize → 0.48
  C: (0.0 + 0.0 + 0.2) / 3 = 0.067 → normalize → 0.08
  Total: 1.0 → normalized to 1.0

Wait, they're the same! Let me check their code again...

Actually, their code does:
  sums[theme] += prob
  cnts[theme] += 1
  dist[theme] = sums[theme] / cnts[theme]

So for C:
  sums[C] = 0.0 + 0.0 + 0.2 = 0.2
  cnts[C] = 1 + 1 + 1 = 3
  dist[C] = 0.2 / 3 = 0.067

But if C only appears in 1 review:
  sums[C] = 0.2
  cnts[C] = 1
  dist[C] = 0.2 / 1 = 0.2

So averaging can give different weights to themes that appear in fewer reviews!
```

### The Real Difference:

**Our Method (Summing):**
- Treats each review equally
- Each review contributes its full probability mass
- Themes that appear in fewer reviews get less total probability (correct)

**Their Method (Averaging):**
- Averages probabilities per theme
- If a theme appears in only 1 review, it gets that review's full probability
- Can over-weight rare themes

### Example Where They Differ:

```
Review 1: {A:0.9, B:0.1}
Review 2: {A:0.8, B:0.2}
Review 3: {A:0.7, B:0.2, C:0.1}  ← C only in this review

Our Method:
  A: 0.9 + 0.8 + 0.7 = 2.4 → 0.857
  B: 0.1 + 0.2 + 0.2 = 0.5 → 0.179
  C: 0.0 + 0.0 + 0.1 = 0.1 → 0.036

Their Method:
  A: (0.9 + 0.8 + 0.7) / 3 = 0.8 → normalize → 0.842
  B: (0.1 + 0.2 + 0.2) / 3 = 0.167 → normalize → 0.175
  C: (0.0 + 0.0 + 0.1) / 1 = 0.1 → normalize → 0.053  ← Different!

Note: C's count is 1 (only appears in 1 review), so it gets 0.1/1 = 0.1
```

---

## Which Method Is Correct?

### For Probability Distributions:
**Our Method (Summing) is more correct** because:
1. Each review is a complete probability distribution (sums to 1.0)
2. Summing preserves the total probability mass
3. Normalizing once gives the correct aggregate distribution
4. Mathematically sound for probability distributions

### Their Method (Averaging) is:
- More like "average probability per theme"
- Can be biased if themes appear in different numbers of reviews
- But simpler to understand conceptually

---

## Summary Table

| Method | What It Measures | When to Use |
|--------|------------------|-------------|
| **Our Approach 1** | Average JSD per review | Understand individual review performance, see variation |
| **Our Approach 2** | Overall distribution similarity | Compare models/methods, get single metric |
| **Their Method** | Average probability distribution | Simpler aggregation, but can be biased |

---

## Recommendation

**Use Our Approach 2 (Summing)** because:
- Mathematically correct for probability distributions
- Treats each review equally
- Standard approach in probability theory
- Matches how distributions should be aggregated

Their averaging method can work, but it's less standard and can introduce bias when themes don't appear uniformly across reviews.

