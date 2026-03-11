# ==========================================
# Movie Recommendation Clustering Project
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import ast  # For safely evaluating string representations of dictionaries
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    file_path = 'movies_metadata.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Dataset '{file_path}' not found.")
        print("Please download 'The Movies Dataset' from Kaggle:")
        print("https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset")
        print("Extract the 'movies_metadata.csv' file into the same directory as this script.")
        return

    # ==========================================
    # 2. Load Dataset
    # ==========================================
    print("\n--- 2. Loading Dataset ---")
    # Low memory=False to handle mixed data types in kaggle dataset
    df = pd.read_csv(file_path, low_memory=False)
    
    print("First 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset Information:")
    print(df.info())
    
    print("\nMissing values before cleaning:")
    print(df[['genres', 'vote_average', 'popularity']].isna().sum())

    # ==========================================
    # 3. Feature Extraction
    # ==========================================
    print("\n--- 3. Feature Extraction ---")
    # We focus on the features requested
    features = ['original_title', 'genres', 'vote_average', 'popularity']
    
    # Create a working copy with only the necessary columns to save memory
    # original_title is kept just for reference later if mapping back
    df_features = df[features].copy()

    # ==========================================
    # 4. Genre Processing
    # ==========================================
    print("\n--- 4. Genre Processing ---")
    
    # The 'genres' column is a string representing a list of dictionaries.
    # We need to parse it safely and extract just the genre names.
    def extract_genres(genre_str):
        try:
            # Safely evaluate the string to a python list
            genre_list = ast.literal_eval(genre_str)
            # Extract names
            return [g['name'] for g in genre_list]
        except (ValueError, SyntaxError, TypeError):
            # Return empty list if parsing fails
            return []

    # Parse and extract
    df_features['genre_list'] = df_features['genres'].apply(extract_genres)
    
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    # Fit and transform the genre list into binary columns
    genre_encoded = pd.DataFrame(mlb.fit_transform(df_features['genre_list']),
                                 columns=mlb.classes_,
                                 index=df_features.index)
                                 
    print(f"Extracted {len(mlb.classes_)} unique genres.")

    # ==========================================
    # 5. Data Cleaning
    # ==========================================
    print("\n--- 5. Data Cleaning ---")
    
    # Convert 'popularity' to numeric, invalid parsing will be set as NaN
    df_features['popularity'] = pd.to_numeric(df_features['popularity'], errors='coerce')
    
    # Combine extracted ratings, popularity, and one-hot encoded genres
    # We drop the old 'genres' and 'genre_list' columns
    df_modeling = pd.concat([df_features[['original_title', 'vote_average', 'popularity']], genre_encoded], axis=1)
    
    # Drop rows with any missing values in the relevant columns
    df_modeling.dropna(subset=['vote_average', 'popularity'], inplace=True)
    
    print(f"Shape of dataset after cleaning missing values: {df_modeling.shape}")

    # For clustering, we need only numeric features (no titles)
    # The original dataset is very large (~45k rows). Hierarchical clustering and TSNE
    # scale poorly (O(N^2) or O(N^3)). We will sample 5000 rows for efficiency if the dataset is too large.
    SAMPLE_SIZE = 5000
    if len(df_modeling) > SAMPLE_SIZE:
        print(f"\nDataset is large ({len(df_modeling)} rows). Sampling {SAMPLE_SIZE} records for efficient clustering and TSNE.")
        df_sample = df_modeling.sample(n=SAMPLE_SIZE, random_state=42).copy()
    else:
        df_sample = df_modeling.copy()
        
    # Isolate clustering features (ratings, popularity + all genre columns)
    X = df_sample.drop(columns=['original_title'])

    # ==========================================
    # 6. Feature Scaling
    # ==========================================
    print("\n--- 6. Feature Scaling ---")
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Normalize the feature values so ratings, popularity, and genres are on a comparable scale
    X_scaled = scaler.fit_transform(X)
    print("Features scaled successfully using StandardScaler.")

    # ==========================================
    # 7. Clustering Models
    # ==========================================
    print("\n--- 7. Clustering Models ---")
    
    # Define number of clusters (arbitrary starting point, could use elbow method)
    n_clusters = 5
    print(f"Number of clusters (K) chosen: {n_clusters}")
    
    # a) KMeans Clustering
    print("Fitting KMeans model...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    df_sample['Cluster_KMeans'] = kmeans_labels
    
    # b) Hierarchical Clustering (AgglomerativeClustering)
    print("Fitting Agglomerative (Hierarchical) model...")
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg_cluster.fit_predict(X_scaled)
    df_sample['Cluster_Hierarchical'] = agg_labels

    # ==========================================
    # 8. Cluster Evaluation
    # ==========================================
    print("\n--- 8. Cluster Evaluation ---")
    
    # Compute Silhouette Score for both models
    # Range is [-1, 1], closer to 1 is better, indicating tight and well-separated clusters
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    agg_silhouette = silhouette_score(X_scaled, agg_labels)
    
    print(f"KMeans Silhouette Score: {kmeans_silhouette:.4f}")
    print(f"Hierarchical Silhouette Score: {agg_silhouette:.4f}")

    # ==========================================
    # 9. Dimensionality Reduction
    # ==========================================
    print("\n--- 9. Dimensionality Reduction ---")
    
    # a) PCA (2 components)
    print("Performing PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_sample['PCA1'] = X_pca[:, 0]
    df_sample['PCA2'] = X_pca[:, 1]
    
    # b) t-SNE (2 components)
    print("Performing t-SNE (this might take a moment)...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    df_sample['TSNE1'] = X_tsne[:, 0]
    df_sample['TSNE2'] = X_tsne[:, 1]

    # ==========================================
    # 10. Visualization
    # ==========================================
    print("\n--- 10. Visualization ---")
    print("Plotting figures...")
    
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color palette
    palette = sns.color_palette("viridis", n_clusters)

    # 1) PCA with KMeans clusters
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_KMeans', data=df_sample, ax=axes[0, 0], palette=palette, alpha=0.7)
    axes[0, 0].set_title('PCA Layout - KMeans Clusters')
    
    # 2) t-SNE with KMeans clusters
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster_KMeans', data=df_sample, ax=axes[0, 1], palette=palette, alpha=0.7)
    axes[0, 1].set_title('t-SNE Layout - KMeans Clusters')
    
    # 3) PCA with Hierarchical clusters
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_Hierarchical', data=df_sample, ax=axes[1, 0], palette=palette, alpha=0.7)
    axes[1, 0].set_title('PCA Layout - Hierarchical Clusters')
    
    # 4) t-SNE with Hierarchical clusters
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster_Hierarchical', data=df_sample, ax=axes[1, 1], palette=palette, alpha=0.7)
    axes[1, 1].set_title('t-SNE Layout - Hierarchical Clusters')

    plt.tight_layout()
    # Save the figure to file
    plt.savefig('cluster_visualizations.png')
    print("Visualizations saved to 'cluster_visualizations.png'.")
    
    # Also attempt to show the plots interactively
    plt.show(block=False)

    # ==========================================
    # 11. Insights
    # ==========================================
    print("\n--- 11. Insights & Cluster Profiling (Based on KMeans) ---")
    
    # Ensure Cluster_KMeans is categorical for cleaner grouping
    df_sample['Cluster_KMeans'] = df_sample['Cluster_KMeans'].astype(int)
    
    # Compute average rating and popularity for each cluster
    cluster_summary = df_sample.groupby('Cluster_KMeans')[['vote_average', 'popularity']].mean()
    
    # Count the number of movies in each cluster
    cluster_counts = df_sample['Cluster_KMeans'].value_counts().sort_index()
    cluster_summary['movie_count'] = cluster_counts
    
    print("\nCluster Summary Statistics (Average Rating, Popularity, and Counts):")
    print(cluster_summary)
    
    # Identify top genres per cluster
    print("\nTop Genres per Cluster:")
    # Group by cluster and mean over the genre boolean columns to get genre frequency
    genre_freq = df_sample.groupby('Cluster_KMeans')[mlb.classes_].mean()
    
    for cluster_id, row in genre_freq.iterrows():
        # Get top 3 genres for this cluster based on frequency
        top_genres = row.nlargest(3).index.tolist()
        # Get frequencies formatted as percentage
        freqs = [f"{val*100:.1f}%" for val in row.nlargest(3).values]
        
        top_genres_str = ", ".join([f"{g} ({f})" for g, f in zip(top_genres, freqs)])
        print(f"Cluster {cluster_id}: {top_genres_str}")

if __name__ == "__main__":
    main()
