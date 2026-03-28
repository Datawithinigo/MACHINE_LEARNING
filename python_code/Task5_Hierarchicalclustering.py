# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering tools
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Reproducibility
np.random.seed(42)
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# Load pre-processed data and setup image output directory
import os
data_dir = '/Users/arriazui/Downloads/master/MACHINE_LEARNING/python_code'
img_dir = os.path.join(data_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

X_cluster = pd.read_csv(os.path.join(data_dir, 'X_cluster.csv'))
X_pca_2d  = pd.read_csv(os.path.join(data_dir, 'X_pca_2d.csv'))

# Sanity check
print(f'X_cluster shape: {X_cluster.shape}')
print(f'X_pca_2d shape:  {X_pca_2d.shape}')

# Computing linkage matrices for two methods
linkage_ward     = linkage(X_cluster, method='ward')
linkage_complete = linkage(X_cluster, method='complete')


print(f'Ward linkage matrix shape:     {linkage_ward.shape}')
print(f'Complete linkage matrix shape: {linkage_complete.shape}')


# drawing the dendogram
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Ward dendrogram
dendrogram(linkage_ward,
           ax=axes[0],
           truncate_mode='lastp',
           p=30, # shows only last 30 merges for readability
           leaf_rotation=90,
           link_color_func=lambda k: 'steelblue')
axes[0].set_title('Dendrogram - Ward linkage')
axes[0].set_xlabel('Patient clusters')
axes[0].set_ylabel('Merge distance')


# Complete dendrogram
dendrogram(linkage_complete,
           ax=axes[1],
           truncate_mode='lastp',
           p=30, # shows only last 30 merges for readability
           leaf_rotation=90,
           link_color_func=lambda k: 'steelblue')
axes[1].set_title('Dendrogram - Complete linkage')
axes[1].set_xlabel('Patient clusters')
axes[1].set_ylabel('Merge distance')


plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task5_01_dendrograms_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# silhoutte scores for Ward
for k in [2, 3, 4]:
    labels = fcluster(linkage_ward, k, criterion='maxclust')
    score  = silhouette_score(X_cluster, labels)
    print(f'k={k} → silhouette score: {score:.4f}')

fig, ax = plt.subplots(figsize=(10, 6))

dendrogram(linkage_ward,
           ax=ax,
           truncate_mode='lastp',
           p=30,
           leaf_rotation=90
           )

ax.set_title('Dendrogram - Ward linkage (selected method)')
ax.set_xlabel('Patient clusters')
ax.set_ylabel('Merge distance')

ax.axhline(y=22, color='red', linestyle='--', linewidth=1.5, label='Cut point (k=2)') # selecting a cut point at height 22
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task5_02_dendrogram_ward_cutpoint.png'), dpi=300, bbox_inches='tight')
plt.show()

# Cutting the Ward dendrogram at k=2
labels_ward = fcluster(linkage_ward, 2, criterion='maxclust')

# Check how many patients ended up in each cluster
unique, counts = np.unique(labels_ward, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f'Cluster {cluster}: {count} patients')

fig, ax = plt.subplots(figsize=(8, 6))

# Define colours for each cluster
colours = {1: 'steelblue', 2: 'darkorange'}
cluster_names = {1: f'Cluster 1 (n=91)', 2: f'Cluster 2 (n=206)'}

for cluster in [1, 2]:
    mask = labels_ward == cluster
    ax.scatter(X_pca_2d.loc[mask, 'PC1'],
               X_pca_2d.loc[mask, 'PC2'],
               c=colours[cluster],
               label=cluster_names[cluster],
               alpha=0.6,
               edgecolors='white',
               linewidths=0.5,
               s=50)

ax.set_title('Hierarchical clustering (Ward, k=2) - PCA projection')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()
plt.tight_layout()
plt.show()

# Add cluster labels to the original feature matrix
X_interpreted = X_cluster.copy()
X_interpreted['cluster'] = labels_ward

# Compute mean of each feature per cluster
cluster_profiles = X_interpreted.groupby('cluster').mean().round(3)
print(cluster_profiles.T)

#plot heatmap for easier visualization
fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(cluster_profiles.T,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            ax=ax)

ax.set_title('Cluster profiles - mean feature values per cluster')
ax.set_xlabel('Cluster')
ax.set_ylabel('Feature')

plt.tight_layout()
plt.show()


