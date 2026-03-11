import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import ast
import os

# ==========================================
# 0. App Configuration
# ==========================================
st.set_page_config(page_title="Movie Clustering Explorer", layout="wide")
st.title("🍿 Movie Clustering Explorer")
st.markdown("Explore and find movie recommendations based on unsupervised Machine Learning clustering (Genres, Rating, and Popularity)!")

# ==========================================
# 1. Helper Functions (Cached for Performance)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    """
    Loads movies_metadata.csv, extracts features, cleans the data, 
    and applies MultiLabelBinarizer to the genres.
    """
    file_path = 'movies_metadata.csv'
    
    if not os.path.exists(file_path):
        st.error(f"Error: Dataset '{file_path}' not found. Please ensure it's in the same directory.")
        st.stop()
        
    # Load dataset
    df = pd.read_csv(file_path, low_memory=False)
    
    # Feature Extraction
    features = ['original_title', 'genres', 'vote_average', 'popularity']
    df_features = df[features].copy()
    
    # Safely parse JSON genre string
    def extract_genres(genre_str):
        try:
            genre_list = ast.literal_eval(genre_str)
            return [g['name'] for g in genre_list]
        except:
            return []
            
    df_features['genre_list'] = df_features['genres'].apply(extract_genres)
    
    # Encode Genres into binary columns
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(mlb.fit_transform(df_features['genre_list']),
                                 columns=mlb.classes_,
                                 index=df_features.index)
                                 
    # Clean Popularity and Ratings
    df_features['popularity'] = pd.to_numeric(df_features['popularity'], errors='coerce')
    df_modeling = pd.concat([df_features[['original_title', 'vote_average', 'popularity', 'genre_list']], genre_encoded], axis=1)
    
    # Drop missing values
    df_modeling.dropna(subset=['vote_average', 'popularity'], inplace=True)
    
    # For Streamlit performance, we sample 5000 movies randomly
    # (Feel free to increase this if your machine can handle it quickly)
    SAMPLE_SIZE = 5000
    if len(df_modeling) > SAMPLE_SIZE:
        df_sample = df_modeling.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        df_sample = df_modeling.reset_index(drop=True)
        
    # Separate the numerical features for the model
    # We drop the string columns: 'original_title' and 'genre_list'
    X = df_sample.drop(columns=['original_title', 'genre_list'])
    
    return df_sample, X, mlb.classes_


@st.cache_data
def run_clustering_and_pca(_X, n_clusters):
    """
    Scales the data, runs KMeans clustering, and applies PCA to 2 components.
    """
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(_X)
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    cluster_centers = kmeans.cluster_centers_
    
    # Dimensionality Reduction (PCA to 2D)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    return cluster_labels, X_pca, cluster_centers, scaler


# ==========================================
# 2. Sidebar UI
# ==========================================
st.sidebar.header("⚙️ Settings")

n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)
min_rating = st.sidebar.slider("Filter: Minimum Rating", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
min_popularity = st.sidebar.slider("Filter: Minimum Popularity", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

# ==========================================
# 3. Data Processing Execution
# ==========================================
with st.spinner("Loading and processing data..."):
    # Run the cached functions
    df_sample, X, genre_classes = load_and_preprocess_data()
    cluster_labels, X_pca, cluster_centers, scaler = run_clustering_and_pca(X, n_clusters)
    
    # Assign new columns to the dataframe
    df_sample['Cluster'] = cluster_labels
    df_sample['PCA1'] = X_pca[:, 0]
    df_sample['PCA2'] = X_pca[:, 1]
    
    # Apply Sidebar Filters
    filtered_df = df_sample[
        (df_sample['vote_average'] >= min_rating) & 
        (df_sample['popularity'] >= min_popularity)
    ]

# ==========================================
# 4. Main UI - Visualizations
# ==========================================
st.write(f"Showing **{len(filtered_df)}** movies out of {len(df_sample)} sampled data points.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("PCA Cluster Visualization")
    st.markdown("This 2D scatter plot represents the high-dimensional movies clustered accurately by their similarities.")
    
    # Plotting using Matplotlib and Seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster', 
        palette=sns.color_palette("viridis", n_clusters),
        data=filtered_df, 
        alpha=0.7, 
        ax=ax,
        legend="full"
    )
    ax.set_title(f'Movie Clusters (K={n_clusters}) projected with PCA')
    st.pyplot(fig)

with col2:
    st.subheader("Cluster Interaction")
    # User selects which cluster to inspect
    selected_cluster = st.selectbox("🎯 Select a Cluster to Explore", options=sorted(filtered_df['Cluster'].unique()))
    
    st.markdown("### Cluster Characteristics")
    # Calculate means for the selected cluster
    cluster_stats = df_sample[df_sample['Cluster'] == selected_cluster]
    st.write(f"**Total Movies in Cluster {selected_cluster}:** {len(cluster_stats)}")
    st.write(f"**Average Rating:** ⭐ {cluster_stats['vote_average'].mean():.2f}")
    st.write(f"**Average Popularity:** 📈 {cluster_stats['popularity'].mean():.2f}")
    
    # Calculate top genres for this cluster
    genre_freq = cluster_stats[genre_classes].mean()
    top_genres = genre_freq.nlargest(3)
    st.write("**Top Genres:**")
    for g, f in top_genres.items():
        st.write(f"- {g} ({f*100:.1f}%)")

# ==========================================
# 5. Recommendations Section
# ==========================================
st.divider()
st.subheader(f"🍿 Top Movie Recommendations for Cluster {selected_cluster}")
st.markdown("These are the highest rated movies that share the characteristics of your selected cluster.")

# Filter by selected cluster, then within the already side-bar filtered dataframe
recommendations = filtered_df[filtered_df['Cluster'] == selected_cluster]

if len(recommendations) > 0:
    # Sort by Rating (Descending) and then Popularity (Descending)
    top_movies = recommendations.sort_values(by=['vote_average', 'popularity'], ascending=[False, False])
    
    # Display the top 10 as a clean table
    display_cols = ['original_title', 'genre_list', 'vote_average', 'popularity']
    
    # Format the genre list into a string for better UI display
    top_movies['Genres'] = top_movies['genre_list'].apply(lambda x: ", ".join(x))
    
    display_df = top_movies[['original_title', 'Genres', 'vote_average', 'popularity']].head(10)
    display_df.columns = ['Title', 'Genres', 'Rating', 'Popularity']
    
    # Use st.dataframe for an interactive table
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No movies found in this cluster matching your minimum sidebar filters. Try lowering the Rating or Popularity threshold!")
    
# ==========================================
# 6. Cluster Centers (Model Weights)
# ==========================================
st.divider()
with st.expander("Show Advanced: Cluster Centers (KMeans Weights)"):
    st.markdown("This table shows the **Standardized** coordinates of the exact center of each predicted cluster in the high-dimensional space.")
    
    # cluster_centers are scaled values. Let's make a pandas DataFrame of the centers.
    feature_names = X.columns
    centers_df = pd.DataFrame(cluster_centers, columns=feature_names)
    centers_df.index.name = "Cluster ID"
    
    st.dataframe(centers_df, use_container_width=True)
