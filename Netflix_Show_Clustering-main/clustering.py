# clustering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(path="data/netflix_cleaned.csv"):
    """Load the cleaned dataset"""
    df = pd.read_csv(path)
    return df

def preprocess_duration(df):
    """Convert duration column to numeric"""
    def convert_duration(x):
        if "Season" in x:
            return int(x.split()[0]) * 10  # Treat 1 season as 10 units
        elif "min" in x:
            return int(x.split()[0])
        else:
            return 0
    df['duration_num'] = df['duration'].apply(convert_duration)
    return df

def encode_features(df):
    """Encode categorical features using One-Hot Encoding"""
    df_encoded = pd.get_dummies(df[['type', 'country', 'rating']], drop_first=True)
    
    # Handle genres (listed_in) - split by comma
    genres = df['listed_in'].str.get_dummies(sep=', ')
    
    # Combine all features
    df_final = pd.concat([df_encoded, genres, df[['release_year', 'duration_num']]], axis=1)
    return df_final

def scale_features(df_features):
    """Normalize numeric features"""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features)
    return df_scaled

def apply_pca(df_scaled, n_components=2, verbose=False):
    """Apply PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df_scaled)
    if verbose:
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    return df_pca

def prepare_clustering_data(path="data/netflix_cleaned.csv"):
    """Full pipeline: load → preprocess → encode → scale → PCA"""
    df = load_data(path)
    df = preprocess_duration(df)
    df_features = encode_features(df)
    df_scaled = scale_features(df_features)
    df_pca = apply_pca(df_scaled)
    return df, df_features, df_scaled, df_pca

if __name__ == "__main__":
    df, df_features, df_scaled, df_pca = prepare_clustering_data()
    print("✅ Data ready for clustering!")

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans_clustering(df_pca, max_clusters=10):
    """Find optimal clusters using Elbow Method and apply K-Means"""
    
    # Elbow Method to find optimal K
    inertia = []
    K_range = range(1, max_clusters+1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_pca)
        inertia.append(kmeans.inertia_)
    
    # Plot Elbow
    plt.figure(figsize=(8,5))
    plt.plot(K_range, inertia, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.show()
    
    # Choose K (you can pick based on elbow, here example K=5)
    optimal_k = 5
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans_final.fit_predict(df_pca)
    
    # Silhouette Score
    score = silhouette_score(df_pca, cluster_labels)
    print(f"Silhouette Score for K={optimal_k}: {score:.4f}")
    
    return cluster_labels, kmeans_final

def plot_clusters(df_pca, cluster_labels):
    """Plot PCA 2D clusters"""
    plt.figure(figsize=(8,6))
    plt.scatter(df_pca[:,0], df_pca[:,1], c=cluster_labels, cmap='tab10', s=50)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Netflix Shows Clusters')
    plt.colorbar(label='Cluster')
    plt.show()

if __name__ == "__main__":
    # Prepare data
    df, df_features, df_scaled, df_pca = prepare_clustering_data()
    print("✅ Data ready for clustering!")

    # K-Means clustering
    labels, kmeans_model = kmeans_clustering(df_pca)
    
    # Visualize clusters
    plot_clusters(df_pca, labels)
