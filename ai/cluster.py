import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import yaml
import os

def load_config():
    """Loads the configuration from settings.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """
    Loads summarized posts, clusters them using K-Means,
    and updates the database with cluster IDs.
    """
    print("--- Starting Clustering Process ---")
    config = load_config()
    db_path = os.path.join(os.path.dirname(__file__), '..', config['database']['path'])
    
    # 1. Load summarized posts from the database using pandas
    try:
        conn = sqlite3.connect(db_path)
        # Select only posts that have a summary but have not yet been clustered
        query = "SELECT id, summary FROM posts WHERE summary IS NOT NULL AND cluster_id IS NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return

    if df.empty or 'summary' not in df.columns or df['summary'].dropna().empty:
        print("No new summaries found to cluster. Exiting.")
        return

    # Ensure we only work with rows that have a valid summary
    df = df.dropna(subset=['summary'])
    print(f"Loaded {len(df)} new summaries from the database.")

    # 2. Vectorize the summaries using TF-IDF
    # This converts the text summaries into a numerical format.
    # We limit to 1000 features to keep it efficient.
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['summary'])
    print("Text vectorization complete.")

    # 3. Apply K-Means clustering
    # We'll try to find 8 distinct topics in the data.
    ai_config = config.get('ai', {})
    num_clusters = ai_config.get('num_clusters', 10) # Default to 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    # Add the cluster labels (0, 1, 2, etc.) to our DataFrame
    df['cluster_id'] = kmeans.labels_
    print(f"Clustering complete. Assigned {len(df)} posts to {num_clusters} clusters.")

    # 4. Save the new cluster IDs back to the database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for index, row in df.iterrows():
            cursor.execute(
                "UPDATE posts SET cluster_id = ? WHERE id = ?",
                (int(row['cluster_id']), row['id'])
            )
        conn.commit()
        conn.close()
        print(f"Successfully saved cluster IDs to the database.")
    except Exception as e:
        print(f"Error saving cluster IDs to database: {e}")
        return

    # 5. Interpret the clusters by finding the top keywords for each
    print("\n--- Top Keywords Per Cluster ---")
    try:
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(num_clusters):
            # Get the top 10 keywords for each cluster
            top_terms = [terms[ind] for ind in order_centroids[i, :10]]
            print(f"Cluster {i}: {', '.join(top_terms)}")
    except Exception as e:
        print(f"Could not print cluster terms: {e}")
        
    print("\n--- Clustering Process Finished ---")

if __name__ == '__main__':
    main()