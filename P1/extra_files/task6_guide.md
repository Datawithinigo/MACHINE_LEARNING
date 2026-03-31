Here's the full guide for Task 6, starting with the pipeline diagram, followed by a second diagram explaining what ARI actually means conceptually.Now here's a conceptual diagram to help you understand what ARI is actually measuring — which matters for the written discussion in Task 7.---

## Task 6 — Full breakdown

### Where your data comes from

You need four things, all already computed in previous tasks: `y` (the original target column you set aside in Task 1), `kmeans_labels`, `gmm_labels`, and `hc_labels`. This task is entirely about connecting those outputs.

---

### Step 1 — Binarise the target

The assignment explicitly tells you to binarise: 0 = no disease, anything 1–4 = disease present.

```python
y_true = (y > 0).astype(int)  # shape (303,) with values 0 or 1
```

Make sure you apply this to the same rows that survived missing-value removal in Task 1 (the Cleveland dataset has ~6 rows with `?` that you should have dropped, leaving 297 patients).

---

### Step 2 — Compute ARI for each method

```python
from sklearn.metrics import adjusted_rand_score

ari_kmeans = adjusted_rand_score(y_true, kmeans_labels)
ari_gmm    = adjusted_rand_score(y_true, gmm_labels)
ari_hc     = adjusted_rand_score(y_true, hc_labels)

import pandas as pd
results = pd.DataFrame({
    'Method': ['K-Means', 'GMM', 'Hierarchical'],
    'ARI':    [ari_kmeans, ari_gmm, ari_hc]
})
print(results.to_string(index=False))
```

ARI is symmetric — `adjusted_rand_score(y_true, labels)` gives the same result as `adjusted_rand_score(labels, y_true)`.

---

### Step 3 — Confusion-style matrices

One matrix per method, three plots in a row (use `plt.subplots(1, 3)`):

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
pairs = [('K-Means', kmeans_labels), ('GMM', gmm_labels), ('Hierarchical', hc_labels)]

for ax, (name, labels) in zip(axes, pairs):
    ct = pd.crosstab(labels, y_true,
                     rownames=['Cluster'], colnames=['True label'])
    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(name)

plt.tight_layout()
```

Important: the rows are cluster IDs (0, 1, ...) and the columns are true labels (0 = no disease, 1 = disease). You're looking for a pattern where one cluster is dominated by one class — that diagonal-like dominance is what a good ARI reflects.

---

### Step 4 — The discussion (where marks are won or lost)

This is worth 6 points just for the matrix + discussion section. A strong answer covers three things:

**1. Which method scored highest and why.** Link the ARI to the algorithm's assumptions. K-Means assumes spherical, equal-sized clusters — if the disease/no-disease groups in this dataset are not spherical or similarly sized, K-Means will struggle. GMM is more flexible (elliptical shapes, varying covariances) so it may do better. Hierarchical clustering with Ward linkage minimises within-cluster variance and often performs well on compact, well-separated groups.

**2. What the confusion matrices reveal.** Look at whether each cluster is "pure" (one class dominates) or "mixed" (both classes roughly equal). A mixed cluster means the algorithm couldn't separate the groups. Point to specific features that help or hurt — e.g., if `thalach` (max heart rate) and `ca` (vessels coloured) are your top selected features, they should be the ones driving separation.

**3. The limits of the comparison.** Unsupervised methods are not optimising for class separation — they're finding geometric structure in the data. So a low ARI doesn't mean the method failed; it means the natural geometric clusters don't perfectly match the clinical diagnosis. This is worth saying explicitly and will stand out in the report.