"""
Project: Implementing and Evaluating K-Means Clustering 
on Synthetic High-Dimensional Data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ------------------------------------------
# Step 1: Generate Synthetic High-Dimensional Data
# ------------------------------------------
n_samples = 1200
n_features = 50
true_clusters = 5

X, y_true = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=true_clusters,
    cluster_std=2.5,
    random_state=42
)

print(f"Dataset shape: {X.shape}")


# ------------------------------------------
# Step 2: Evaluate K-Means for K=2 to 10
# ------------------------------------------
inertia_values = []
silhouette_scores = []

K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

    print(f"K={k} | Inertia={kmeans.inertia_:.2f} | "
          f"Silhouette Score={silhouette_scores[-1]:.4f}")


# ------------------------------------------
# Step 3: Choose Optimal K (Highest Silhouette Score)
# ------------------------------------------
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal K based on Silhouette Score: {optimal_k}")


# ------------------------------------------
# Step 4: PCA for 2D Visualization
# ------------------------------------------
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, s=10)
plt.title(f"PCA Projection of Clusters (K={optimal_k})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()
