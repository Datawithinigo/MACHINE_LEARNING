import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
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

# ── Sweep k = 2 to 10, recording inertia and silhouette score ─────────────
# Both metrics together guide the selection of the optimal k.

k_range    = range(2, 11)
inertias   = []
sil_scores = []

for k in k_range:
    km = KMeans(
        n_clusters=k,
        init='k-means++',  # Smart seeding - reduces risk of poor local minima
        n_init=20,         # 20 independent runs, best result is kept
        random_state=42
    )
    labels = km.fit_predict(X_cluster)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster, labels))
    print(f'k={k:<2d}  |  Inertia: {km.inertia_:8.2f}  |  Silhouette: {silhouette_score(X_cluster, labels):.4f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: Elbow (inertia) ──────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(list(k_range), inertias, marker='o', linewidth=2,
         color=PALETTE[0], markersize=8, markerfacecolor='white',
         markeredgewidth=2, markeredgecolor=PALETTE[0])
ax1.axvline(x=2, color=PALETTE[1], linestyle='--', linewidth=1.5, alpha=0.8,
            label='Candidate k=2')
ax1.set_xlabel('Number of Clusters k', fontsize=11)
ax1.set_ylabel('Inertia (Within-Cluster SSE)', fontsize=11)
ax1.set_title('Elbow Method', fontsize=13, fontweight='bold')
ax1.set_xticks(list(k_range))
ax1.legend()

# ── Right: Silhouette Score ────────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(list(k_range), sil_scores, marker='s', linewidth=2,
         color=PALETTE[2], markersize=8, markerfacecolor='white',
         markeredgewidth=2, markeredgecolor=PALETTE[2])
best_k_sil = list(k_range)[np.argmax(sil_scores)]
ax2.axvline(x=best_k_sil, color=PALETTE[1], linestyle='--', linewidth=1.5,
            alpha=0.8, label=f'Best k={best_k_sil}')
ax2.set_xlabel('Number of Clusters k', fontsize=11)
ax2.set_ylabel('Silhouette Score', fontsize=11)
ax2.set_title('Silhouette Score per k', fontsize=13, fontweight='bold')
ax2.set_xticks(list(k_range))
ax2.legend()

plt.suptitle('K-Means: Elbow Method and Silhouette Score', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task3_01_elbow_silhouette.png'), dpi=300, bbox_inches='tight')
#  plt.show()

# ── Fit final K-Means with chosen k ───────────────────────────────────────
k_opt = 2   # Supported by both elbow inflection and maximum silhouette score

km_final = KMeans(
    n_clusters=k_opt,
    init='k-means++',   # Smart initialisation: reduces risk of poor local minima
    n_init=20,          # Run 20 independent initialisations, keep best result
    random_state=42     # Reproducibility
)
cluster_labels = km_final.fit_predict(X_cluster)

print(f'Final K-Means with k={k_opt}')
print(f'Inertia         : {km_final.inertia_:.4f}')
print(f'Silhouette Score: {silhouette_score(X_cluster, cluster_labels):.4f}')

# Cluster size distribution - check for degenerate (very unequal) splits
print(f'\nCluster sizes:')
unique, counts = np.unique(cluster_labels, return_counts=True)
for c, n in zip(unique, counts):
    print(f'  Cluster {c}: {n} patients ({n/len(cluster_labels)*100:.1f}%)')

# Project cluster centres into the same PCA 2D space for visual reference
pca_2d_model = PCA(n_components=2, random_state=42).fit(X_cluster)
centers_pca  = pca_2d_model.transform(km_final.cluster_centers_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cluster_palette = [PALETTE[i] for i in range(k_opt)]

# ── Left: K-Means cluster assignments ──────────────────────────────────────
ax1 = axes[0]
for c in range(k_opt):
    mask = cluster_labels == c
    ax1.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                label=f'Cluster {c} (n={mask.sum()})',
                color=cluster_palette[c], alpha=0.6, s=30,
                edgecolors='white', linewidths=0.4)

# Cluster centres as bold X markers
for c in range(k_opt):
    ax1.scatter(centers_pca[c, 0], centers_pca[c, 1],
                marker='X', s=180, color=cluster_palette[c],
                edgecolors='black', linewidths=1.2, zorder=5,
                label=f'Centre {c}')

ax1.set_xlabel('PC1 (27.6% variance)', fontsize=11)
ax1.set_ylabel('PC2 (14.0% variance)', fontsize=11)
ax1.set_title(f'K-Means Clusters (k={k_opt}) in PCA Space',
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

plt.suptitle('K-Means Cluster Assignments vs. Ground Truth in PCA Space',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task3_02_clusters_vs_groundtruth.png'), dpi=300, bbox_inches='tight')
#  plt.show()

# ── Cluster centres in the original feature space ─────────────────────────
centers_df = pd.DataFrame(km_final.cluster_centers_, columns=X_cluster.columns)
centers_df.index = [f'Cluster {i}' for i in range(k_opt)]

# ── Heatmap of cluster centres ─────────────────────────────────────────────
# Red = above population mean (z > 0), Blue = below (z < 0)
fig, ax = plt.subplots(figsize=(14, 3.5))
sns.heatmap(
    centers_df,
    annot=True, fmt='.2f', cmap=PALETTE_DIV,
    center=0, linewidths=0.5, linecolor='white',
    ax=ax, annot_kws={'size': 8}
)
ax.set_title(f'K-Means Cluster Centres (k={k_opt}) - Scaled Feature Space',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Feature', fontsize=11)
ax.set_ylabel('Cluster', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task3_03_cluster_centres_heatmap.png'), dpi=300, bbox_inches='tight')
#  plt.show()

# ── Top and bottom features per cluster (ranked by z-score) ───────────────
print('=== Top 5 features (highest z-score) per cluster ===')
for c in range(k_opt):
    top = centers_df.loc[f'Cluster {c}'].nlargest(5)
    print(f'\nCluster {c}:')
    for feat, val in top.items():
        print(f'  {feat:<18} {val:+.3f}')

print('\n=== Bottom 5 features (lowest z-score) per cluster ===')
for c in range(k_opt):
    bot = centers_df.loc[f'Cluster {c}'].nsmallest(5)
    print(f'\nCluster {c}:')
    for feat, val in bot.items():
        print(f'  {feat:<18} {val:+.3f}')
