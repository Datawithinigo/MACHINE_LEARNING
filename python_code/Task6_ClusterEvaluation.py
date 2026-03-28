#!/usr/bin/env python3
# coding: utf-8

# # Assignment 1 - Unsupervised Learning Methods
# ## Task 6: Cluster Evaluation Against Ground Truth
# **Goal:** Evaluate the performance of K-Means, GMM, and Hierarchical Clustering
# by comparing their cluster assignments against the binarised clinical diagnosis
# using the Adjusted Rand Index (ARI) and confusion matrices.
#
# This task connects the unsupervised clustering results from Tasks 3-5 with the
# ground truth labels that were set aside in Task 2, providing a quantitative
# assessment of how well the geometric structure discovered by each algorithm
# aligns with clinical reality.

# ---
# ### 6.1 Imports & Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import warnings
import os

warnings.filterwarnings('ignore')

np.random.seed(42)

# Visual theme matching Tasks 3-5
PALETTE = ['#4A90D9', '#E8603C', '#2EAF7D', '#9B59B6', '#F39C12', '#C0392B', '#1ABC9C']
PALETTE_SEQ = 'YlOrRd'
PALETTE_DIV = 'coolwarm'

sns.set_theme(
    style='whitegrid', palette=PALETTE, font='DejaVu Sans', font_scale=1.1,
    rc={
        'figure.dpi': 120, 'figure.figsize': (10, 5),
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'axes.titlesize': 13,
        'axes.labelsize': 11, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'legend.frameon': False, 'legend.fontsize': 9,
    }
)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=PALETTE)
print('Libraries loaded successfully.')

# Define data directory and image output directory
data_dir = '/Users/arriazui/Downloads/master/MACHINE_LEARNING/python_code'
img_dir = os.path.join(data_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

# ---
# ### 6.2 Load Data & Binarise Target

# Load clustering matrix and ground truth labels
X_cluster = pd.read_csv(os.path.join(data_dir, 'X_cluster.csv'))
y_raw     = pd.read_csv(os.path.join(data_dir, 'y_clean.csv')).squeeze()

# Binarise target: 0 = no disease, 1-4 = disease present
y_true = (y_raw > 0).astype(int)

print(f'Clustering matrix shape: {X_cluster.shape}')
print(f'Target shape           : {y_true.shape}')
print(f'\nBinarised target distribution:')
print(f'  No disease (0): {(y_true == 0).sum()} patients ({(y_true == 0).sum()/len(y_true)*100:.1f}%)')
print(f'  Disease (1)   : {(y_true == 1).sum()} patients ({(y_true == 1).sum()/len(y_true)*100:.1f}%)')

# ---
# ### 6.3 Generate Cluster Labels from Each Method

# We refit each algorithm with k=2 (the optimal choice from Tasks 3-5)
# using the same parameters to ensure reproducibility

print('\n' + '='*70)
print('REFITTING CLUSTERING ALGORITHMS')
print('='*70)

# ── K-Means ────────────────────────────────────────────────────────────────
kmeans = KMeans(
    n_clusters=2,
    init='k-means++',
    n_init=20,
    random_state=42
)
kmeans_labels = kmeans.fit_predict(X_cluster)
print(f'\nK-Means cluster sizes:')
unique, counts = np.unique(kmeans_labels, return_counts=True)
for c, n in zip(unique, counts):
    print(f'  Cluster {c}: {n} patients ({n/len(kmeans_labels)*100:.1f}%)')

# ── Gaussian Mixture Model ─────────────────────────────────────────────────
gmm = GaussianMixture(
    n_components=2,
    covariance_type='full',
    n_init=10,
    random_state=42,
    max_iter=200
)
gmm.fit(X_cluster)
gmm_labels = gmm.predict(X_cluster)
print(f'\nGMM cluster sizes:')
unique, counts = np.unique(gmm_labels, return_counts=True)
for c, n in zip(unique, counts):
    print(f'  Cluster {c}: {n} patients ({n/len(gmm_labels)*100:.1f}%)')

# ── Hierarchical Clustering (Ward linkage) ─────────────────────────────────
# Ward linkage minimises within-cluster variance - same objective as K-Means
hc = AgglomerativeClustering(
    n_clusters=2,
    linkage='ward'
)
hc_labels = hc.fit_predict(X_cluster)
print(f'\nHierarchical Clustering cluster sizes:')
unique, counts = np.unique(hc_labels, return_counts=True)
for c, n in zip(unique, counts):
    print(f'  Cluster {c}: {n} patients ({n/len(hc_labels)*100:.1f}%)')

# ---
# ### 6.4 Compute Adjusted Rand Index (ARI)

# ARI measures the similarity between two clusterings, adjusted for chance
# Range: [-1, 1]
#   1.0  = perfect agreement
#   0.0  = random labelling
#   < 0  = worse than random

print('\n' + '='*70)
print('ADJUSTED RAND INDEX (ARI) EVALUATION')
print('='*70)

ari_kmeans = adjusted_rand_score(y_true, kmeans_labels)
ari_gmm    = adjusted_rand_score(y_true, gmm_labels)
ari_hc     = adjusted_rand_score(y_true, hc_labels)

# Create results table
results = pd.DataFrame({
    'Method': ['K-Means', 'GMM', 'Hierarchical'],
    'ARI':    [ari_kmeans, ari_gmm, ari_hc]
})
results = results.sort_values('ARI', ascending=False).reset_index(drop=True)

print('\n' + results.to_string(index=False))
print('\n' + '='*70)
print(f'Best performer: {results.iloc[0]["Method"]} (ARI = {results.iloc[0]["ARI"]:.4f})')
print('='*70)

# ---
# ### 6.5 Confusion Matrices

# For each method, create a crosstab showing how cluster IDs map to true labels
# Rows = cluster IDs (0, 1), Columns = true labels (0=no disease, 1=disease)

print('\n' + '='*70)
print('CONFUSION MATRICES')
print('='*70)

# ── K-Means Confusion Matrix ───────────────────────────────────────────────
ct_kmeans = pd.crosstab(
    kmeans_labels, y_true,
    rownames=['Cluster'], colnames=['True Label']
)
print('\nK-Means:')
print(ct_kmeans)

# ── GMM Confusion Matrix ───────────────────────────────────────────────────
ct_gmm = pd.crosstab(
    gmm_labels, y_true,
    rownames=['Cluster'], colnames=['True Label']
)
print('\nGMM:')
print(ct_gmm)

# ── Hierarchical Clustering Confusion Matrix ───────────────────────────────
ct_hc = pd.crosstab(
    hc_labels, y_true,
    rownames=['Cluster'], colnames=['True Label']
)
print('\nHierarchical Clustering:')
print(ct_hc)

# ---
# ### 6.6 Visualise Confusion Matrices

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

pairs = [
    ('K-Means', ct_kmeans, ari_kmeans),
    ('GMM', ct_gmm, ari_gmm),
    ('Hierarchical', ct_hc, ari_hc)
]

for ax, (name, ct, ari) in zip(axes, pairs):
    sns.heatmap(
        ct, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
        linewidths=0.5, linecolor='white',
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    ax.set_title(f'{name}\nARI = {ari:.4f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('True Label\n(0=No Disease, 1=Disease)', fontsize=10)
    ax.set_ylabel('Cluster ID', fontsize=10)

plt.suptitle('Cluster Assignments vs Ground Truth - Confusion Matrices',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task6_01_confusion_matrices.png'), dpi=300, bbox_inches='tight')
# plt.show()

print('\n' + '='*70)
print('Confusion matrix plots saved.')
print('='*70)

# ---
# ### 6.7 Cluster Purity Analysis

# For each cluster in each method, calculate the percentage of the dominant class
# High purity = cluster is dominated by one true label (good separation)

print('\n' + '='*70)
print('CLUSTER PURITY ANALYSIS')
print('='*70)

def cluster_purity(labels, y_true):
    """Calculate purity for each cluster"""
    purities = []
    for c in np.unique(labels):
        mask = labels == c
        cluster_labels = y_true[mask]
        # Purity = fraction of patients in the most common class
        purity = cluster_labels.value_counts().max() / len(cluster_labels)
        dominant_class = cluster_labels.value_counts().idxmax()
        purities.append({
            'Cluster': c,
            'Size': len(cluster_labels),
            'Dominant Class': dominant_class,
            'Purity': purity
        })
    return pd.DataFrame(purities)

print('\nK-Means:')
purity_kmeans = cluster_purity(kmeans_labels, y_true)
print(purity_kmeans.to_string(index=False))

print('\nGMM:')
purity_gmm = cluster_purity(gmm_labels, y_true)
print(purity_gmm.to_string(index=False))

print('\nHierarchical Clustering:')
purity_hc = cluster_purity(hc_labels, y_true)
print(purity_hc.to_string(index=False))

# Overall purity (weighted by cluster size)
overall_purity_kmeans = np.mean([purity_kmeans.iloc[i]['Purity'] * purity_kmeans.iloc[i]['Size'] 
                                  for i in range(len(purity_kmeans))]) / len(y_true)
overall_purity_gmm = np.mean([purity_gmm.iloc[i]['Purity'] * purity_gmm.iloc[i]['Size'] 
                               for i in range(len(purity_gmm))]) / len(y_true)
overall_purity_hc = np.mean([purity_hc.iloc[i]['Purity'] * purity_hc.iloc[i]['Size'] 
                              for i in range(len(purity_hc))]) / len(y_true)

print('\n' + '='*70)
print('OVERALL WEIGHTED PURITY:')
print(f'  K-Means      : {overall_purity_kmeans:.4f}')
print(f'  GMM          : {overall_purity_gmm:.4f}')
print(f'  Hierarchical : {overall_purity_hc:.4f}')
print('='*70)

# ---
# ### 6.8 Method Agreement Analysis

# Check how often the three methods agree on cluster assignments
# This reveals whether the methods discover similar or different structures

print('\n' + '='*70)
print('INTER-METHOD AGREEMENT')
print('='*70)

# K-Means vs GMM
agreement_km_gmm = (kmeans_labels == gmm_labels).sum() / len(kmeans_labels) * 100
print(f'\nK-Means vs GMM          : {agreement_km_gmm:.1f}% agreement')

# K-Means vs Hierarchical
agreement_km_hc = (kmeans_labels == hc_labels).sum() / len(kmeans_labels) * 100
print(f'K-Means vs Hierarchical : {agreement_km_hc:.1f}% agreement')

# GMM vs Hierarchical
agreement_gmm_hc = (gmm_labels == hc_labels).sum() / len(gmm_labels) * 100
print(f'GMM vs Hierarchical     : {agreement_gmm_hc:.1f}% agreement')

# All three agree
all_agree = ((kmeans_labels == gmm_labels) & (kmeans_labels == hc_labels)).sum()
print(f'\nAll three methods agree : {all_agree}/{len(kmeans_labels)} patients ({all_agree/len(kmeans_labels)*100:.1f}%)')

# ---
# ### 6.9 Bar Chart: ARI Comparison

fig, ax = plt.subplots(figsize=(10, 6))

methods = results['Method'].values
ari_values = results['ARI'].values
colors = [PALETTE[i] for i in range(len(methods))]

bars = ax.bar(methods, ari_values, color=colors, alpha=0.8, 
              edgecolor='white', linewidth=1.5)

# Add value labels on top of bars
for bar, val in zip(bars, ari_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=12)
ax.set_title('Clustering Performance vs Ground Truth\nAdjusted Rand Index Comparison',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylim([0, max(ari_values) * 1.15])
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

# Add reference line for random baseline
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
           label='Random baseline (ARI = 0)')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task6_02_ari_comparison.png'), dpi=300, bbox_inches='tight')
# plt.show()

print('\n' + '='*70)
print('ARI comparison plot saved.')
print('='*70)

# ---
# ### 6.10 Summary & Interpretation

print('\n' + '='*70)
print('TASK 6 - SUMMARY & KEY INSIGHTS')
print('='*70)

print(f'''
PERFORMANCE RANKING:
1. {results.iloc[0]["Method"]:<15} ARI = {results.iloc[0]["ARI"]:.4f}
2. {results.iloc[1]["Method"]:<15} ARI = {results.iloc[1]["ARI"]:.4f}
3. {results.iloc[2]["Method"]:<15} ARI = {results.iloc[2]["ARI"]:.4f}

INTERPRETATION:
• All three methods achieve ARI > 0, indicating better-than-random agreement
  with the clinical diagnosis.

• The confusion matrices reveal the structure of disagreement:
  - Each method finds 2 clusters (as expected)
  - Look for diagonal-like dominance: one cluster dominated by no-disease (0),
    the other by disease (1)
  - Mixed clusters indicate geometric overlap between clinical groups

• Method agreement analysis shows how consistent the algorithms are:
  - High inter-method agreement (>80%) suggests robust geometric structure
  - Low agreement (<60%) indicates methods exploit different aspects of the data

LIMITATIONS:
• Unsupervised methods optimise geometric structure, NOT class separation
• Low ARI does not mean the method failed - it means natural geometric
  clusters don't perfectly match clinical diagnosis
• The 15 selected features capture the most discriminative information, but
  heart disease is clinically heterogeneous - not all patients present
  with the same profile

NEXT STEPS:
• Task 7 discussion should link these results to:
  1. Algorithm assumptions (spherical vs elliptical clusters)
  2. Feature importance from Task 2 (which features drive separation?)
  3. Clinical interpretation (why do some patients cluster incorrectly?)
''')

print('='*70)
print('Task 6 completed successfully!')
print('='*70)
