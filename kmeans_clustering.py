import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Synthetic Dataset
# -----------------------------

np.random.seed(42)

n_samples = 500
n_per_cluster = n_samples // 3

# Create 3 non-linearly separable clusters (curved / circular pattern)

# Cluster 1 (circle)
theta1 = np.random.uniform(0, 2*np.pi, n_per_cluster)
r1 = np.random.normal(5, 0.5, n_per_cluster)
x1 = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]

# Cluster 2 (shifted circle)
theta2 = np.random.uniform(0, 2*np.pi, n_per_cluster)
r2 = np.random.normal(10, 0.5, n_per_cluster)
x2 = np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)]

# Cluster 3 (shifted blob)
x3 = np.random.randn(n_per_cluster, 2) + np.array([15, 0])

# Combine dataset
X = np.vstack((x1, x2, x3))
true_labels = np.array([0]*n_per_cluster + 
                       [1]*n_per_cluster + 
                       [2]*n_per_cluster)

# ---------------------------------
# 2. K-Means Implementation (NumPy)
# ---------------------------------

def initialize_centroids(X, K):
    indices = np.random.choice(len(X), K, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, K):
    return np.array([X[labels == k].mean(axis=0) for k in range(K)])

def compute_inertia(X, labels, centroids):
    inertia = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        inertia += np.sum((cluster_points - centroids[k])**2)
    return inertia

def kmeans(X, K, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, K)
    
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, K)
        
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        
        centroids = new_centroids
    
    inertia = compute_inertia(X, labels, centroids)
    return labels, centroids, inertia

# ---------------------------------
# 3. Elbow Method (K = 2 to 10)
# ---------------------------------

inertia_values = []
K_range = range(2, 11)

for K in K_range:
    labels, centroids, inertia = kmeans(X, K)
    inertia_values.append(inertia)

# Plot Elbow Curve
plt.figure()
plt.plot(K_range, inertia_values, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# ---------------------------------
# 4. Silhouette Score (NumPy only)
# ---------------------------------

def silhouette_score(X, labels):
    n = len(X)
    unique_labels = np.unique(labels)
    silhouette_vals = []
    
    for i in range(n):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == k] for k in unique_labels if k != labels[i]]
        
        # Mean intra-cluster distance
        a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
        
        # Mean nearest-cluster distance
        b = min([np.mean(np.linalg.norm(cluster - X[i], axis=1)) 
                 for cluster in other_clusters])
        
        silhouette_vals.append((b - a) / max(a, b))
    
    return np.mean(silhouette_vals)

# Choose optimal K (from elbow visually assumed K=3)
optimal_K = 3
labels_opt, centroids_opt, inertia_opt = kmeans(X, optimal_K)

sil_score = silhouette_score(X, labels_opt)

print("Final Centroids:\n", centroids_opt)
print("Final Inertia:", inertia_opt)
print("Silhouette Score:", sil_score)

# ---------------------------------
# 5. Final Cluster Visualization
# ---------------------------------

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels_opt)
plt.scatter(centroids_opt[:, 0], centroids_opt[:, 1], marker='X', s=200)
plt.title("Final Clusters with Centroids")
plt.show()
    
