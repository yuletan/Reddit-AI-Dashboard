# dashboard/app.py

import streamlit as st
import pandas as pd
import sqlite3
import os

# --- Page Configuration ---
# This should be the first Streamlit command in your script
st.set_page_config(
    page_title="Reddit AI Dashboard",
    page_icon="🧠",
    layout="wide",
)

# --- Data Loading ---
# Use Streamlit's caching to load the data once and reuse it.
@st.cache_data
def load_data():
    """Connects to the SQLite database and loads the posts table into a pandas DataFrame."""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reddit_data.db')
    try:
        conn = sqlite3.connect(db_path)
        # Load only posts that have been summarized and clustered
        query = "SELECT * FROM posts WHERE summary IS NOT NULL AND cluster_id IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to load data from database: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error

# --- Main Application ---
def main():
    st.title("🧠 Reddit AI Dashboard")
    st.markdown("An interactive dashboard to explore AI-summarized and clustered Reddit discussions.")

    df_posts = load_data()

    if df_posts.empty:
        st.warning("No data found. Please run the scraper and clustering scripts first.")
        return

    # --- NEW: Sidebar for Filters ---
    st.sidebar.header("Filters")
    
    # Get a list of unique subreddits from the data
    # We add an 'All' option to see the full dataset
    subreddit_list = ['All'] + sorted(df_posts['subreddit'].unique().tolist())
    
    selected_subreddit = st.sidebar.selectbox(
        "Select a Subreddit",
        subreddit_list
    )

    # --- Filter the DataFrame based on selection ---
    if selected_subreddit == 'All':
        filtered_df = df_posts
    else:
        filtered_df = df_posts[df_posts['subreddit'] == selected_subreddit]
    
    st.success(f"Displaying {len(filtered_df)} posts for subreddit: '{selected_subreddit}'")
    
    # Display the FILTERED data
    st.subheader("Filtered Data View")
    st.dataframe(filtered_df)


if __name__ == "__main__":
    main()