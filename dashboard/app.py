# dashboard/app.py
import streamlit as st
import pandas as pd
import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px 


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
def get_cluster_keywords(df: pd.DataFrame):
    """Analyzes the summaries to find the top keywords for each cluster."""
    keywords_dict = {}
    if 'cluster_id' not in df.columns or df['cluster_id'].dropna().empty:
        return keywords_dict

    # Ensure summaries are strings
    df['summary'] = df['summary'].astype(str)
    
    unique_clusters = sorted(df['cluster_id'].unique())
    
    for cluster_id in unique_clusters:
        # Filter summaries for the current cluster
        cluster_summaries = df[df['cluster_id'] == cluster_id]['summary']
        
        if cluster_summaries.empty:
            continue
            
        # Vectorize the text to find top terms
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        vectorizer.fit_transform(cluster_summaries)
        
        # Get the top 10 keywords
        top_terms = vectorizer.get_feature_names_out()[:10]
        keywords_dict[cluster_id] = ", ".join(top_terms)
        
    return keywords_dict


@st.cache_data
def load_data():
    """Connects to the SQLite database and loads the posts table into a pandas DataFrame."""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reddit_data.db')
    try:
        conn = sqlite3.connect(db_path)
        # Load only posts that have been summarized AND clustered
        query = "SELECT * FROM posts WHERE summary IS NOT NULL AND cluster_id IS NOT NULL AND summary != 'NoSummaryGenerated'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
        return df
    except Exception as e:
        st.error(f"Failed to load data from database: {e}")
        return pd.DataFrame()

# --- Main Application ---
def main():
    st.title("🧠 Reddit AI Dashboard")
    st.markdown("An interactive dashboard to explore AI-summarized and clustered Reddit discussions.")
    
    df_posts = load_data()

    if df_posts.empty:
        st.warning("No data found. Please run the scraper and clustering scripts first.")
        return

    # --- Sidebar for Filters ---
    st.sidebar.header("Filters")

    # Subreddit Filter
    subreddit_list = sorted(df_posts['subreddit'].unique().tolist())
    selected_subreddits = st.sidebar.multiselect("Select Subreddits", subreddit_list, default=subreddit_list)

    # Cluster Multi-Select Filter 
    cluster_list = sorted(df_posts['cluster_id'].unique().tolist()) 
    selected_clusters = st.sidebar.multiselect("Select Topic Clusters", cluster_list, default=cluster_list)
        
        # Date Range Filter 
    min_date = df_posts['created_utc'].min().date() 
    max_date = df_posts['created_utc'].max().date() 
    selected_date_range = st.sidebar.date_input( 
        "Select Date Range", 
        value=(min_date, max_date), 
        min_value=min_date, 
        max_value=max_date 
    ) 

    # Score Range Filter 
    min_score = int(df_posts['score'].min()) 
    max_score = int(df_posts['score'].max()) 
    selected_score_range = st.sidebar.slider( 
        "Select Score Range", 
        min_value=min_score, 
        max_value=max_score, 
        value=(min_score, max_score) 
    ) 

    search_term = st.sidebar.text_input("Search in Summaries")
    # --- Filter the DataFrame based on selections ---

    filtered_df = df_posts

    # Apply filters sequentially
    if selected_subreddits:
        filtered_df = filtered_df[filtered_df['subreddit'].isin(selected_subreddits)]
    if selected_clusters:
        filtered_df = filtered_df[filtered_df['cluster_id'].isin(selected_clusters)]


    # Ensure date range has two values before trying to filter
    if len(selected_date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['created_utc'].dt.date >= selected_date_range[0]) & 
            (filtered_df['created_utc'].dt.date <= selected_date_range[1])
        ]

    filtered_df = filtered_df[
        (filtered_df['score'] >= selected_score_range[0]) & 
        (filtered_df['score'] <= selected_score_range[1])
    ]

    if search_term:
        filtered_df = filtered_df[filtered_df['summary'].str.contains(search_term, case=False, na=False)]

    st.header("Key Metrics")
    total_posts = len(filtered_df)
    average_score = round(filtered_df['score'].mean(), 2) if not filtered_df.empty else 0
    average_sentiment = round(filtered_df['sentiment'].mean(), 2) if 'sentiment' in filtered_df.columns and not filtered_df.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Matching Posts", total_posts)
    col2.metric("Average Score", f"{average_score:,.0f}")
    col3.metric("Average Sentiment", f"{average_sentiment:.2f}")
    st.divider()

    # Create Tabs for Content
    tab1, tab2, tab3 = st.tabs(["💬 Discussion Summaries", "📊 Visualizations", "📋 Raw Data"])

    with tab1:
        st.header("Discussion Summaries")
        st.info(f"Displaying {len(filtered_df)} posts matching your filters.")
        sorted_df = filtered_df.sort_values(by='score', ascending=False)

        for index, post in sorted_df.iterrows():
            st.subheader(f"[{post['title']}]({post['url']})")
            st.caption(f"r/{post['subreddit']} • Cluster: {post['cluster_id']} • Score: {post['score']} • {post['created_utc'].strftime('%Y-%m-%d')}")
            with st.container(border=True):
                st.markdown("**🤖 AI-Generated Summary:**")
                st.write(post['summary'])
            st.divider()

    with tab2:
        st.header("Visualizations")
        if not filtered_df.empty:
            st.subheader("Distribution of Sentiment Scores")
            fig_sentiment = px.histogram(filtered_df, x="sentiment", nbins=50, title="Distribution of Sentiment Scores")
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.warning("No data to visualize for the current filter selection.")

    with tab3:
        st.header("Raw Data Table")
        st.dataframe(filtered_df)


if __name__ == "__main__":
    main()