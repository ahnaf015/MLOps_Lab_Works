"""
clustering.py
-------------
Agglomerative (Hierarchical) Clustering on the Air Quality dataset.

Steps:
  1. Find optimal number of clusters (k=2..6) using silhouette score
  2. Fit final AgglomerativeClustering model with Ward linkage
  3. Generate:
       - Dendrogram (scipy, sampled for readability)
       - PCA 2-D cluster scatter plot
  Both plots are saved as a single PNG to working_data/.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')            
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

WORKING_DATA = '/opt/airflow/working_data'


# Task 1: Train Agglomerative Clustering 

def train_agglomerative(**context):
    """
    Sweep k=2..6 clusters, pick the k with the highest silhouette score,
    then refit the final model and store results.
    """
    split_path = context['ti'].xcom_pull(task_ids='preprocessing_group.split_data')

    with open(split_path, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']

    print("Sweeping cluster counts (k=2..6) by silhouette score...")
    best_k, best_score = 2, -1.0

    for k in range(2, 7):
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X_train)
        score = silhouette_score(X_train, labels, sample_size=1000, random_state=42)
        print(f"  k={k}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"\nBest k={best_k}  silhouette={best_score:.4f}")

    final_model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    labels = final_model.fit_predict(X_train)

    cluster_sizes = {f"Cluster {i}": int(cnt)
                     for i, cnt in enumerate(np.bincount(labels))}
    print(f"Cluster sizes: {cluster_sizes}")

    cluster_path = os.path.join(WORKING_DATA, 'clustering.pkl')
    with open(cluster_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'labels': labels,
            'best_k': best_k,
            'silhouette_score': best_score,
            'cluster_sizes': cluster_sizes,
            'X_train': X_train,
        }, f)

    return cluster_path


# Task 2: Generate Dendrogram + PCA Plot 

def generate_cluster_plots(**context):
    """
    Produce a side-by-side figure:
      Left  — Ward linkage dendrogram (200-point sample, truncated)
      Right — PCA 2-D cluster scatter
    Saved to working_data/dendrogram.png
    """
    cluster_path = context['ti'].xcom_pull(
        task_ids='clustering_group.train_agglomerative'
    )

    with open(cluster_path, 'rb') as f:
        data = pickle.load(f)

    X = data['X_train']
    labels = data['labels']
    best_k = data['best_k']
    sil = data['silhouette_score']

    # Sample for dendrogram 
    sample_n = min(200, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=sample_n, replace=False)

    X_np = X.values if hasattr(X, 'values') else np.array(X)
    X_sample = X_np[idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Air Quality — Agglomerative Clustering  |  k={best_k}  |  '
        f'Silhouette={sil:.3f}',
        fontsize=14, fontweight='bold'
    )

    # Left: Dendrogram 
    Z = linkage(X_sample, method='ward')
    dendrogram(
        Z, ax=axes[0],
        truncate_mode='lastp', p=25,
        leaf_rotation=90, leaf_font_size=8,
        show_contracted=True,
        color_threshold=0.7 * max(Z[:, 2]),
    )
    axes[0].set_title(f'Ward Linkage Dendrogram\n(sample of {sample_n} points)',
                      fontsize=12)
    axes[0].set_xlabel('Sample index / cluster size')
    axes[0].set_ylabel('Ward Distance')

    # Right: PCA scatter 
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_np)

    palette = plt.cm.Set2(np.linspace(0, 0.8, best_k))
    for i in range(best_k):
        mask = labels == i
        axes[1].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[palette[i]], label=f'Cluster {i + 1}',
            alpha=0.55, s=15, edgecolors='none'
        )

    axes[1].set_title(
        f'Cluster Projection (PCA)\n'
        f'PC1={pca.explained_variance_ratio_[0]:.1%}  '
        f'PC2={pca.explained_variance_ratio_[1]:.1%}',
        fontsize=12
    )
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend(markerscale=2)

    plt.tight_layout()

    plot_path = os.path.join(WORKING_DATA, 'dendrogram.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Cluster plots saved → {plot_path}")
    return plot_path
