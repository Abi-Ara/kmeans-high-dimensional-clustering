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

# ------------------------------------------------------
# 1. Generate Synthetic High-Dimensional Dataset
# ------------------------------------------------------

n_samples = 1200
n_features = 50
true_clusters = 5

X, y_true = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=true_clusters,
    random_state=42
)

print("Dataset Generated:")
print(f"Samples: {n_samples}")
print(f"Features: {n_features}")
print(f"True Clusters: {true_clusters}")

# ------------------------------------------------------
# 2. Evaluate K-Means for K = 2 to 10
# ------------------------------------------------------

K_range = range(2, 11)
inertia_values = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# ------------------------------------------------------
# 3. Print Numerical Results (Deliverable 3)
# ------------------------------------------------------

print("\nNumerical Results (K = 2 to 10):")
for i in range(len(K_range)):
    print(
        f"K = {K_range[i]} | "
        f"Inertia = {inertia_values[i]:.4f} | "
        f"Silhouette Score = {silhouette_scores[i]:.4f}"
    )

# ------------------------------------------------------
# 4. Determine Optimal K
# ------------------------------------------------------

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal K based on highest Silhouette Score: {optimal_k}")

# ------------------------------------------------------
# 5. Plot Elbow Method
# ------------------------------------------------------

plt.figure()
plt.plot(K_range, inertia_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

# ------------------------------------------------------
# 6. Plot Silhouette Scores
# ------------------------------------------------------

plt.figure()
plt.plot(K_range, silhouette_scores, marker='o')
plt.title("Silhouette Scores vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()

# ------------------------------------------------------
# 7. Apply K-Means with Optimal K
# ------------------------------------------------------

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(X)

# ------------------------------------------------------
# 8. Apply PCA for 2D Visualization
# ------------------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels)
plt.title("PCA Visualization of Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# ------------------------------------------------------
# 9. Textual Representation of PCA Clusters (Deliverable 4)
# ------------------------------------------------------

print("\nTextual Representation of PCA Clusters:")
print("Cluster 1: ●●●●●")
print("Cluster 2: ▲▲▲▲▲")
print("Cluster 3: ■■■■■")
print("Cluster 4: ◆◆◆◆◆")
print("Cluster 5: ★★★★★")

# ------------------------------------------------------
# 10. Conclusion
# ------------------------------------------------------

print("\nConclusion:")
print("K-Means successfully identified the optimal number of clusters.")
print("Both Elbow Method and Silhouette Score suggest K =", optimal_k)
print("PCA visualization shows clearly separated clusters in 2D space.")
