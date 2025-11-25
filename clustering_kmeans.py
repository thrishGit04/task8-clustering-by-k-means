# Task 8 – K-Means Clustering (Full Pipeline)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams["figure.figsize"] = (7, 5)

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
df = pd.read_csv("Housing.csv") 
print("First 5 rows:")
display(df.head())

# Keep only numeric columns for clustering
df_numeric = df.select_dtypes(include=[np.number]).dropna()

print("\nNumeric columns used for clustering:")
print(df_numeric.columns.tolist())

# ------------------------------------------
# 2. Scale the Features
# ------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ------------------------------------------
# 3. Optional PCA (2D Projection for Visualization)
# ------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Save for plotting
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

# ------------------------------------------
# 4. Elbow Method to find optimal K
# ------------------------------------------
inertia_list = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)

# Plot Elbow Curve
plt.figure()
plt.plot(K_range, inertia_list, marker="o")
plt.title("Elbow Method – Optimal K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Pick K = 3 by default (you can change after viewing elbow plot)
K = 3

# ------------------------------------------
# 5. Fit K-Means with chosen K
# ------------------------------------------
kmeans = KMeans(n_clusters=K, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to PCA dataframe
pca_df["Cluster"] = labels

# ------------------------------------------
# 6. Silhouette Score
# ------------------------------------------
sil_score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", sil_score)

# ------------------------------------------
# 7. Visualize Clusters (2D PCA)
# ------------------------------------------
plt.figure(figsize=(8, 6))
for c in range(K):
    plt.scatter(
        pca_df[pca_df["Cluster"] == c]["PC1"],
        pca_df[pca_df["Cluster"] == c]["PC2"],
        label=f"Cluster {c}", s=40
    )

plt.title("K-Means Clusters (PCA 2D View)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------
# Results
# ------------------------------------------
print("\nCluster Centers (original feature space):")
display(pd.DataFrame(kmeans.cluster_centers_, columns=df_numeric.columns))

print("\nTask 8 – K-Means Clustering completed successfully!")
