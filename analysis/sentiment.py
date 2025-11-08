# analysis/sentiment.py

import sqlite3
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import yaml

def load_config():
    """Loads the configuration from settings.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """
    Analyzes the sentiment of post summaries and updates the database.
    """
    print("--- Starting Sentiment Analysis Process ---")
    config = load_config()
    db_path = os.path.join(os.path.dirname(__file__), '..', config['database']['path'])

    # 1. Load posts that need sentiment analysis
    try:
        conn = sqlite3.connect(db_path)
        # Select posts that have a valid summary but DO NOT have a sentiment score yet.
        # This makes the script incremental - it won't re-process old posts.
        query = "SELECT id, summary FROM posts WHERE summary IS NOT NULL AND summary != 'NoSummaryGenerated' AND sentiment IS NULL"
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return

    if df.empty:
        print("No new summaries found to analyze. Exiting.")
        return

    print(f"Found {len(df)} new summaries to analyze for sentiment.")

    # 2. Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # 3. Calculate sentiment for each summary
    # VADER returns a dictionary with 'neg', 'neu', 'pos', and 'compound' scores.
    # The 'compound' score is a single, normalized value from -1 (most negative) to +1 (most positive).
    # We will use the compound score for our analysis.
    df['sentiment'] = df['summary'].apply(lambda summary: analyzer.polarity_scores(summary)['compound'])

    print("Sentiment calculation complete.")

    # 4. Save the new sentiment scores back to the database
    try:
        # We use a bulk update method for efficiency
        updates = df[['sentiment', 'id']].to_records(index=False).tolist()
        
        cursor = conn.cursor()
        cursor.executemany("UPDATE posts SET sentiment = ? WHERE id = ?", updates)
        conn.commit()
        conn.close()
        print(f"Successfully saved {len(updates)} sentiment scores to the database.")
    except Exception as e:
        print(f"Error saving sentiment scores to the database: {e}")
        return

    print("\n--- Sentiment Analysis Finished ---")

if __name__ == '__main__':
    main()