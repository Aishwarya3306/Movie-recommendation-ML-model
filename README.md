# Movie Recommendation Clustering

This repository contains a full Machine Learning pipeline to group movies into clusters based on their genres, user ratings, and popularity metrics using "The Movies Dataset" from Kaggle.

## Project Structure

*   `movie_clustering.py`: Python script containing the full ML pipeline.
*   `README.md`: Setup and usage instructions.

## Pipeline Overview

1.  **Data Loading**: Loads the dataset and inspects for missing values.
2.  **Feature Engineering & Preprocessing**:
    *   Extracts genre names from JSON formatted strings in the `genres` column.
    *   Applies one-hot encoding for the genres using `MultiLabelBinarizer`.
    *   Cleans and converts `popularity` to a numeric format.
    *   Handles missing data by dropping incomplete rows.
3.  **Scaling**: Standardizes all features (genres, `vote_average`, `popularity`) using `StandardScaler`.
4.  **Clustering**: Applies two different clustering algorithms to group the movies:
    *   **KMeans** clustering.
    *   **Hierarchical (Agglomerative)** clustering.
5.  **Evaluation**: Computes the **Silhouette Score** to evaluate cluster tightness and separation.
6.  **Dimensionality Reduction**: Shrinks the highly dimensional feature space to 2D for visualization using:
    *   **PCA** (Principal Component Analysis)
    *   **t-SNE** (t-distributed Stochastic Neighbor Embedding)
7.  **Visualization & Profiling**: Generates a 2x2 grid of scatter plots illustrating the clusters using Seaborn and Matplotlib. It also extracts insights by calculating average ratings, popularity, and identifying the dominant genres within each KMeans cluster.

## Setup Instructions

### 1. Requirements

Ensure you have Python installed. Install the necessary libraries using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Download the Dataset

The script requires the Kaggle dataset **The Movies Dataset**.

1. Go to the dataset page: [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
2. Download the archive and extract it.
3. Move `movies_metadata.csv` to the same folder where `movie_clustering.py` is located.

### 3. Running the Pipeline

Open your terminal or command prompt, navigate to the folder, and run:

```bash
python movie_clustering.py
```

### Output

*   **Console Output**: The script will print out steps of the pipeline, first rows of the dataset, missing values, clustering progress, silhouette scores, and finally, a summary profile for each found cluster.
*   **Visualizations**: A pop-up window will display a 2x2 grid `cluster_visualizations.png` showing the projection of KMeans and Hierarchical clusters onto the 2D plane using PCA and t-SNE. The figure will also be saved locally as `cluster_visualizations.png`.
