# Wasserstein Distance Calculation: Comparison

## Their Approach (Aggregate-Level Comparison)

### How P and Q are determined:

**P (Source Distribution):**
- Aggregates ALL reviews from a source (e.g., `pre_sgo_train`)
- For each topic in `GROUND_TRUTH_TOPICS`, calculates:
  - Sum of probabilities across all reviews
  - Average = sum / count
- Creates ONE distribution representing the entire dataset

**Q (Target Distribution):**
- Same process for target source (e.g., `real_train`)
- Creates ONE distribution representing the entire dataset

### Calculation Steps:

1. **Build Ground Truth Topic Set:**
   ```python
   GROUND_TRUTH_TOPICS = set()
   for review in ALL_REVIEWS['real_train']:
       topics = review.get('theme_token_probabilities', {})
       GROUND_TRUTH_TOPICS.update(topics.keys())
   ```

2. **Build Aggregate Distributions:**
   ```python
   def build_distribution(reviews, source_type):
       topic_sums = defaultdict(float)
       topic_counts = defaultdict(int)
       
       for review in reviews:
           themes = review.get('predicted_themes', {})
           for topic, prob in themes.items():
               if topic in GROUND_TRUTH_TOPICS:
                   topic_sums[topic] += float(prob)
                   topic_counts[topic] += 1
       
       # Average probability per topic
       distribution = {}
       for topic in GROUND_TRUTH_TOPICS:
           if topic_counts[topic] > 0:
               distribution[topic] = topic_sums[topic] / topic_counts[topic]
           else:
               distribution[topic] = 0.0
       
       # Normalize to sum to 1
       distribution = {k: v/total for k, v in distribution.items()}
   ```

3. **Create Semantic Cost Matrix:**
   ```python
   # Embed topics using sentence transformers
   topic_embeddings = embedder.encode(GROUND_TRUTH_TOPICS)
   
   # Cost = cosine distance between embeddings
   cost_matrix = cosine_distances(topic_embeddings)
   # cost_matrix[i,j] = 1 - cosine_similarity(embedding_i, embedding_j)
   ```

4. **Calculate WD:**
   ```python
   def calculate_wd(dist1, dist2):
       # Align to ground truth topics
       vec1 = np.array([dist1.get(t, 0.0) for t in GROUND_TRUTH_TOPICS])
       vec2 = np.array([dist2.get(t, 0.0) for t in GROUND_TRUTH_TOPICS])
       
       # Normalize
       vec1 = (vec1 + epsilon) / (vec1 + epsilon).sum()
       vec2 = (vec2 + epsilon) / (vec2 + epsilon).sum()
       
       # WD with semantic cost matrix
       wd = ot.emd2(vec1, vec2, cost_matrix)
   ```

### Key Characteristics:
- **Aggregation Level:** Dataset-level (one WD per dataset pair)
- **Cost Matrix:** Semantic (cosine distance between topic embeddings)
- **Topic Alignment:** Fixed ground truth topic set
- **Purpose:** Compare overall distribution of entire datasets

---

## Our Approach (Per-Review Comparison)

### How P and Q are determined:

**P (Source Distribution):**
- For EACH individual review:
  - Gets `prediction.predicted_themes` (dict of theme → probability)
  - Normalizes to sum to 1.0

**Q (Target Distribution):**
- For the SAME review:
  - Gets `actual.predicted_themes` (dict of theme → probability)
  - Normalizes to sum to 1.0

### Calculation Steps:

1. **Align Distributions Per Review:**
   ```python
   def align_distributions(actual_themes, predicted_themes):
       # Get all unique themes from BOTH distributions (dynamic)
       all_themes = sorted(set(actual_themes.keys()) | set(predicted_themes.keys()))
       
       # Create aligned arrays
       actual_array = np.array([actual_themes.get(theme, 0.0) for theme in all_themes])
       predicted_array = np.array([predicted_themes.get(theme, 0.0) for theme in all_themes])
       
       # Normalize
       actual_array = actual_array / actual_array.sum()
       predicted_array = predicted_array / predicted_array.sum()
   ```

2. **Create Unit Cost Matrix:**
   ```python
   # Unit cost: all themes equally distant
   cost_matrix = np.ones((n, n)) - np.eye(n)
   # cost_matrix[i,j] = 1 if i != j, else 0
   ```

3. **Calculate WD:**
   ```python
   def calculate_wasserstein_1_distance(actual_array, predicted_array):
       P = predicted_array / predicted_array.sum()  # Predicted (source)
       Q = actual_array / actual_array.sum()        # Actual (target)
       
       # Unit cost matrix
       cost_matrix = np.ones((n, n)) - np.eye(n)
       
       # WD with unit cost
       wd = ot.emd2(P, Q, cost_matrix)
   ```

### Key Characteristics:
- **Aggregation Level:** Review-level (one WD per review)
- **Cost Matrix:** Unit cost (C[i,j] = 1 if i != j, else 0)
- **Topic Alignment:** Dynamic (uses all unique themes from both distributions)
- **Purpose:** Measure per-review prediction accuracy

---

## Key Differences Summary

| Aspect | Their Approach | Our Approach |
|--------|---------------|--------------|
| **P (Source)** | Aggregate distribution across all reviews in dataset | Individual review's predicted themes |
| **Q (Target)** | Aggregate distribution across all reviews in dataset | Individual review's actual themes |
| **Aggregation** | Dataset-level (one WD per dataset pair) | Review-level (one WD per review) |
| **Cost Matrix** | Semantic (cosine distance from embeddings) | Unit cost (all themes equally distant) |
| **Topic Set** | Fixed ground truth topics | Dynamic (unique themes per review) |
| **Use Case** | Compare overall dataset distributions | Measure per-review accuracy |

---

## Mathematical Formulation

Both use the same formula:
```
W_1(P, Q) = min_γ Σ_i Σ_j γ_ij * C_ij
```

Where:
- **P** = source probability distribution
- **Q** = target probability distribution  
- **γ** = optimal transport plan (matrix)
- **C_ij** = cost matrix (different in each approach)

### Their Cost Matrix:
```
C_ij = 1 - cosine_similarity(embedding_i, embedding_j)
```
- Similar themes (e.g., "Comfort" vs "Fit") have lower cost
- Captures semantic relationships

### Our Cost Matrix:
```
C_ij = 1 if i != j, else 0
```
- All themes equally distant
- Simpler, treats all mismatches equally

---

## Which is Correct?

**Both are mathematically correct!** They serve different purposes:

1. **Their approach:** Good for comparing overall dataset distributions
   - Answers: "How similar are the overall topic distributions between datasets?"
   - Uses semantic cost to account for theme similarity

2. **Our approach:** Good for measuring per-review prediction accuracy
   - Answers: "How accurate is the prediction for this specific review?"
   - Uses unit cost for categorical comparison

The choice depends on your goal:
- **Dataset comparison** → Their approach
- **Per-review accuracy** → Our approach
















