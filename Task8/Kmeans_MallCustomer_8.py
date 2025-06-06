import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

#  Step 2: Environment Setup (Fix MKL memory warning on Windows)

os.environ["OMP_NUM_THREADS"] = "1"


# Step 3: Load Dataset
df = pd.read_csv(r"C:\Users\capl2\OneDrive\Pictures\Documents\AIML_Internship\Task8\Mall_Customers.csv") 
print("Dataset Loaded\n")

# Step 4: Initial Inspection
print(df.head())

# Step 5: Preprocessing
# Drop irrelevant column
df.drop("CustomerID", axis=1, inplace=True)

# Encode 'Gender' (Male/Female -> 1/0)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#  Step 6: PCA for 2D Visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(6, 5))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c='gray', s=60)
plt.title("Unclustered Data (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

# Step 7: Elbow Method to Find Optimal K
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(6, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

# Step 8: Apply KMeans with Optimal K
optimal_k = 5  # Based on elbow curve
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to DataFrame
df['Cluster'] = cluster_labels

# Step 9: Visualize Clusters using PCA
plt.figure(figsize=(7, 5))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=cluster_labels, palette='Set2', s=100)
plt.title('Customer Segments (K-Means, PCA Projection)')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

#  Step 10: Evaluate with Silhouette Score
score = silhouette_score(scaled_data, cluster_labels)
print(f"\n Silhoutte score for={optimal_k}: {score:.3f}")

# Step 11: Display Sample Clustered Data
print("\nSample Clustered Data:")
print(df.head())
