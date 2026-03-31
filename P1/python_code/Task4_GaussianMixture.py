import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import joblib

warnings.filterwarnings('ignore')

np.random.seed(42)

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
import os
data_dir = '/Users/arriazui/Downloads/master/MACHINE_LEARNING/python_code'
img_dir = os.path.join(data_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

X_cluster = pd.read_csv(os.path.join(data_dir, 'X_cluster.csv'))
X_pca_2d  = pd.read_csv(os.path.join(data_dir, 'X_pca_2d.csv')).values   # shape (297, 2)
y         = pd.read_csv(os.path.join(data_dir, 'y_clean.csv')).squeeze()
# load SAME PCA used in preprocessing
pca_2d_model = joblib.load(os.path.join(data_dir, "pca_2d_model.pkl"))
print(f'Clustering matrix : {X_cluster.shape}')
print(f'PCA 2D projection : {X_pca_2d.shape}')
print(f'Target labels     : {y.shape}')
print(f'\nFeatures for clustering:\n{list(X_cluster.columns)}')

# ── Step 1: Fit GMMs for k = 2 to 10, recording BIC and AIC ───────────────
# BIC and AIC are model selection criteria that balance fit and complexity
# Lower values indicate better models

k_range    = range(2, 11)
bic_scores = []
aic_scores = []
gmm_models = []

for k in k_range:
    gm = GaussianMixture(
        n_components=k,
        covariance_type='full',  # Full covariance allows elliptical clusters
        n_init=10,               # 10 independent runs to avoid poor local optima
        random_state=42,
        max_iter=200
    )
    gm.fit(X_cluster)
    bic_scores.append(gm.bic(X_cluster))
    aic_scores.append(gm.aic(X_cluster))
    gmm_models.append(gm)
    print(f'k={k:<2d}  |  BIC: {gm.bic(X_cluster):8.2f}  |  AIC: {gm.aic(X_cluster):8.2f}')

# ── Step 2: Plot BIC and AIC curves ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: BIC (Bayesian Information Criterion) ─────────────────────────────
# Lower BIC = better model. BIC penalizes complexity more than AIC.
ax1 = axes[0]
ax1.plot(list(k_range), bic_scores, marker='o', linewidth=2,
         color=PALETTE[0], markersize=8, markerfacecolor='white',
         markeredgewidth=2, markeredgecolor=PALETTE[0])
best_k_bic = list(k_range)[np.argmin(bic_scores)]
ax1.axvline(x=best_k_bic, color=PALETTE[1], linestyle='--', linewidth=1.5,
            alpha=0.8, label=f'Best k={best_k_bic} (min BIC)')
ax1.set_xlabel('Number of Components k', fontsize=11)
ax1.set_ylabel('BIC Score', fontsize=11)
ax1.set_title('Bayesian Information Criterion', fontsize=13, fontweight='bold')
ax1.set_xticks(list(k_range))
ax1.legend()

# ── Right: AIC (Akaike Information Criterion) ──────────────────────────────
# Lower AIC = better model. AIC penalizes complexity less than BIC.
ax2 = axes[1]
ax2.plot(list(k_range), aic_scores, marker='s', linewidth=2,
         color=PALETTE[2], markersize=8, markerfacecolor='white',
         markeredgewidth=2, markeredgecolor=PALETTE[2])
best_k_aic = list(k_range)[np.argmin(aic_scores)]
ax2.axvline(x=best_k_aic, color=PALETTE[1], linestyle='--', linewidth=1.5,
            alpha=0.8, label=f'Best k={best_k_aic} (min AIC)')
ax2.set_xlabel('Number of Components k', fontsize=11)
ax2.set_ylabel('AIC Score', fontsize=11)
ax2.set_title('Akaike Information Criterion', fontsize=13, fontweight='bold')
ax2.set_xticks(list(k_range))
ax2.legend()

plt.suptitle('Gaussian Mixture Model: BIC and AIC Model Selection',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task4_01_bic_aic.png'), dpi=300, bbox_inches='tight')
#  plt.show()

# ── Step 3: Refit the final GMM with chosen k ─────────────────────────────
k_opt = 2   # BIC/AIC differences, we selected k=2 as optimal

gmm_final = GaussianMixture(
    n_components=k_opt,
    covariance_type='full',  # Full covariance: elliptical clusters
    n_init=10,               # 10 independent initialisations
    random_state=42,
    max_iter=200
)
gmm_final.fit(X_cluster)
gmm_labels = gmm_final.predict(X_cluster)
gmm_proba  = gmm_final.predict_proba(X_cluster)  # Soft assignments

print(f'\nFinal GMM with k={k_opt}')
print(f'BIC Score       : {gmm_final.bic(X_cluster):.4f}')
print(f'AIC Score       : {gmm_final.aic(X_cluster):.4f}')
print(f'Log Likelihood  : {gmm_final.score(X_cluster) * len(X_cluster):.4f}')

# Cluster size distribution
print(f'\nCluster sizes:')
unique, counts = np.unique(gmm_labels, return_counts=True)
for c, n in zip(unique, counts):
    print(f'  Cluster {c}: {n} patients ({n/len(gmm_labels)*100:.1f}%)')

# Average certainty of assignments (mean of max probability per patient)
certainties = gmm_proba.max(axis=1)
print(f'\nAssignment certainty (mean max probability): {certainties.mean():.3f}')
print(f'  Min: {certainties.min():.3f}  |  Max: {certainties.max():.3f}')

# ── Step 4: PCA scatter plot ───────────────────────────────────────────────
# Project GMM cluster means into the same PCA 2D space for visual reference
pca_2d_model = PCA(n_components=2, random_state=42).fit(X_cluster)
means_pca = pca_2d_model.transform(gmm_final.means_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cluster_palette = [PALETTE[i] for i in range(k_opt)]

# ── Left: GMM cluster assignments ──────────────────────────────────────────
ax1 = axes[0]
for c in range(k_opt):
    mask = gmm_labels == c
    ax1.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                label=f'Cluster {c} (n={mask.sum()})',
                color=cluster_palette[c], alpha=0.6, s=30,
                edgecolors='white', linewidths=0.4)

# Cluster means as bold X markers
for c in range(k_opt):
    ax1.scatter(means_pca[c, 0], means_pca[c, 1],
                marker='X', s=180, color=cluster_palette[c],
                edgecolors='black', linewidths=1.2, zorder=5,
                label=f'Mean {c}')

ax1.set_xlabel('PC1 (27.6% variance)', fontsize=11)
ax1.set_ylabel('PC2 (14.0% variance)', fontsize=11)
ax1.set_title(f'GMM Clusters (k={k_opt}) in PCA Space',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=8)

# ── Right: Ground truth overlay (reference only) ───────────────────────────
ax2 = axes[1]
gt_palette = {0: '#4CAF50', 1: '#F44336'}
gt_labels  = {0: 'No Disease', 1: 'Disease'}
for label in [0, 1]:
    mask = y == label
    ax2.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                label=gt_labels[label], color=gt_palette[label],
                alpha=0.6, s=30, edgecolors='white', linewidths=0.4)

ax2.set_xlabel('PC1 (27.6% variance)', fontsize=11)
ax2.set_ylabel('PC2 (14.0% variance)', fontsize=11)
ax2.set_title('Ground Truth (Reference Only)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)

plt.suptitle('GMM Cluster Assignments vs. Ground Truth in PCA Space',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task4_02_clusters_vs_groundtruth.png'), dpi=300, bbox_inches='tight')
#  plt.show()

# ── Step 5: Characterise clusters in original feature space ───────────────
means_df = pd.DataFrame(gmm_final.means_, columns=X_cluster.columns)
means_df.index = [f'Cluster {i}' for i in range(k_opt)]

# ── Heatmap of cluster means ───────────────────────────────────────────────
# Red = above population mean (z > 0), Blue = below (z < 0)
fig, ax = plt.subplots(figsize=(14, 3.5))
sns.heatmap(
    means_df,
    annot=True, fmt='.2f', cmap=PALETTE_DIV,
    center=0, linewidths=0.5, linecolor='white',
    ax=ax, annot_kws={'size': 8}
)
ax.set_title(f'GMM Cluster Means (k={k_opt}) - Scaled Feature Space',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Feature', fontsize=11)
ax.set_ylabel('Cluster', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task4_03_cluster_means_heatmap.png'), dpi=300, bbox_inches='tight')
#  plt.show()

# ── Top and bottom features per cluster (ranked by z-score) ───────────────
print('\n=== Top 5 features (highest z-score) per cluster ===')
for c in range(k_opt):
    top = means_df.loc[f'Cluster {c}'].nlargest(5)
    print(f'\nCluster {c}:')
    for feat, val in top.items():
        print(f'  {feat:<18} {val:+.3f}')

print('\n=== Bottom 5 features (lowest z-score) per cluster ===')
for c in range(k_opt):
    bot = means_df.loc[f'Cluster {c}'].nsmallest(5)
    print(f'\nCluster {c}:')
    for feat, val in bot.items():
        print(f'  {feat:<18} {val:+.3f}')

# ── Step 6: Compare with K-Means ───────────────────────────────────────────
print('\n' + '='*60)
print('COMPARISON: GMM vs K-Means')
print('='*60)

# Load K-Means results from Task 3
km_final = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
kmeans_labels = km_final.fit_predict(X_cluster)

# Crosstab to show agreement/disagreement
comparison = pd.crosstab(
    kmeans_labels, gmm_labels,
    rownames=['K-Means'], colnames=['GMM'],
    margins=True
)
print('\nCluster assignment crosstab:')
print(comparison)

# Agreement percentage
agreement = (kmeans_labels == gmm_labels).sum() / len(kmeans_labels) * 100
print(f'\nAgreement: {agreement:.1f}% of patients assigned to same cluster')

# Where do they disagree?
disagreement_mask = kmeans_labels != gmm_labels
print(f'\nDisagreement: {disagreement_mask.sum()} patients ({100-agreement:.1f}%)')

if disagreement_mask.sum() > 0:
    # Check assignment certainty for disagreement cases
    disagreement_certainty = certainties[disagreement_mask].mean()
    agreement_certainty = certainties[~disagreement_mask].mean()
    print(f'\nAverage GMM certainty where methods AGREE:    {agreement_certainty:.3f}')
    print(f'Average GMM certainty where methods DISAGREE: {disagreement_certainty:.3f}')
    print('\n→ GMM soft probabilities reveal boundary cases where K-Means may be forcing')
    print('  hard assignments on ambiguous patients.')

# ── Visualise disagreement in PCA space ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Plot agreement cases in light colors
agreement_mask = ~disagreement_mask
for c in range(k_opt):
    mask = (gmm_labels == c) & agreement_mask
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               label=f'Agreement - Cluster {c}',
               color=cluster_palette[c], alpha=0.3, s=30,
               edgecolors='white', linewidths=0.3)

# Highlight disagreement cases
ax.scatter(X_pca_2d[disagreement_mask, 0], X_pca_2d[disagreement_mask, 1],
           label='K-Means ≠ GMM',
           color='black', alpha=0.8, s=50, marker='^',
           edgecolors='yellow', linewidths=1.5)

ax.set_xlabel('PC1 (27.6% variance)', fontsize=11)
ax.set_ylabel('PC2 (14.0% variance)', fontsize=11)
ax.set_title('GMM vs K-Means: Agreement and Disagreement in PCA Space',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task4_04_gmm_vs_kmeans.png'), dpi=300, bbox_inches='tight')
#  plt.show()

print('\n' + '='*60)
print('Task 4 completed successfully!')
print('='*60)
print('\nKey insights:')
print('• GMM is a probabilistic model with soft cluster assignments')
print('• Full covariance allows elliptical (non-spherical) clusters')
print('• BIC/AIC balance model fit against complexity')
print('• GMM provides assignment probabilities → clinically valuable')
print('• Comparison with K-Means reveals boundary patients')
