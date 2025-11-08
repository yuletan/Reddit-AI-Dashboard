import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import argparse
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
    Can run in two modes:
    1. Default (Incremental): Only clusters new posts.
    2. Recluster (--recluster): Wipes all old clusters and re-clusters everything.
    """
    ### NEW - Set up the argument parser
    parser = argparse.ArgumentParser(description="Cluster Reddit post summaries.")
    parser.add_argument(
        '--recluster',
        action='store_true',  # This makes it a flag, e.g., `python cluster.py --recluster`
        help="Clear all existing cluster IDs and recluster all posts from scratch."
    )
    args = parser.parse_args()

    print("--- Starting Clustering Process ---")
    config = load_config()
    db_path = os.path.join(os.path.dirname(__file__), '..', config['database']['path'])
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    ### NEW - Logic to handle the --recluster flag
    if args.recluster:
        print("!!! RECLUSTER MODE ENABLED !!!")
        print("    -> Clearing all existing cluster IDs from the database...")
        try:
            # This is the crucial step: reset all cluster_id's to NULL
            cursor.execute("UPDATE posts SET cluster_id = NULL")
            conn.commit()
            print("    -> All cluster IDs have been reset.")
        except Exception as e:
            print(f"Error resetting cluster IDs: {e}")
            conn.close()
            return
        
        # In recluster mode, we select ALL posts that have a summary
        query = "SELECT id, summary FROM posts WHERE summary IS NOT NULL AND summary != 'NoSummaryGenerated'"
        print("    -> Loading ALL summarized posts for reclustering.")
    else:
        # This is the original, incremental behavior
        print("--- INCREMENTAL MODE ---")
        query = "SELECT id, summary FROM posts WHERE summary IS NOT NULL AND summary != 'NoSummaryGenerated' AND cluster_id IS NULL"
        print("    -> Loading only new, unclustered posts.")

    # 1. Load summarized posts from the database using pandas
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error loading data from database: {e}")
        conn.close()
        return
    finally:
        # We're done with the initial connection for now
        conn.close()

    if df.empty or 'summary' not in df.columns or df['summary'].dropna().empty:
        if args.recluster:
            print("No summaries found in the database to cluster. Exiting.")
        else:
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