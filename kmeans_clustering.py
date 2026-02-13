"""
Implementing and Evaluating K-Means Clustering from Scratch

This script:
1. Generates synthetic data using make_blobs
2. Implements K-Means using only NumPy
3. Uses the Elbow Method (WCSS) to determine optimal K
4. Compares results with sklearn KMeans
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# ==========================================================
# 1. Generate Synthetic Dataset
# ==========================================================

np.random.seed(42)

X, y_true = make_blobs(
    n_samples=500,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

print("Dataset shape:", X.shape)

# ==========================================================
# 2. Custom K-Means Implementation (NumPy Only)
# ==========================================================

def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def compute_distances(X, centroids):
    # Vectorized Euclidean distance
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

def assign_clusters(distances):
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        new_centroids.append(cluster_points.mean(axis=0))
    return np.array(new_centroids)

def compute_wcss(X, centroids, labels):
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        wcss += np.sum((cluster_points - centroids[i]) ** 2)
    return wcss

def kmeans_custom(X, k, max_iters=300, tol=1e-4):
    centroids = initialize_centroids(X, k)

    for iteration in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = assign_clusters(distances)
        new_centroids = update_centroids(X, labels, k)

        shift = np.linalg.norm(new_centroids - centroids)

        if shift < tol:
            break

        centroids = new_centroids

    wcss = compute_wcss(X, centroids, labels)

    return centroids, labels, wcss


# ==========================================================
# 3. Elbow Method (K = 2 to 7)
# ==========================================================

print("\nWCSS Values for K = 2 to 7:")

wcss_values = []
K_range = list(range(2, 8))

for k in K_range:
    _, _, wcss = kmeans_custom(X, k)
    wcss_values.append(wcss)
    print(f"K = {k}, WCSS = {wcss:.2f}")

# Programmatic elbow detection using second derivative
wcss_array = np.array(wcss_values)
second_derivative = np.diff(wcss_array, 2)
optimal_index = np.argmax(-second_derivative) + 1
optimal_k = K_range[optimal_index]

print("\nOptimal K detected by Elbow Method:", optimal_k)


# ==========================================================
# 4. Compare With sklearn KMeans
# ==========================================================

# Custom KMeans using optimal K
custom_centroids, custom_labels, custom_wcss = kmeans_custom(X, optimal_k)

# Sklearn KMeans
sklearn_kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=20
)

sklearn_kmeans.fit(X)

sklearn_centroids = sklearn_kmeans.cluster_centers_
sklearn_labels = sklearn_kmeans.labels_

# Sort centroids for fair comparison
custom_sorted = np.sort(custom_centroids, axis=0)
sklearn_sorted = np.sort(sklearn_centroids, axis=0)

centroid_difference = np.abs(custom_sorted - sklearn_sorted)

print("\nCustom Centroids:\n", custom_sorted)
print("\nSklearn Centroids:\n", sklearn_sorted)
print("\nAbsolute Centroid Differences:\n", centroid_difference)

# Cluster similarity
ari_score = adjusted_rand_score(custom_labels, sklearn_labels)

print("\nAdjusted Rand Index (Cluster Agreement):", ari_score)

print("\nCustom WCSS:", custom_wcss)
print("Sklearn Inertia:", sklearn_kmeans.inertia_)
