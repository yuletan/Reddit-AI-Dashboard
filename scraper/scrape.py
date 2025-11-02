from ai.summarize import summarize_text
from ai.summarize import summarize_batch, fake_summarize_batch
import praw
import yaml
import sqlite3
import os
import time 

def load_config():
    """Loads the configuration from settings.yaml."""
    # Correct the path to be relative to the current file's location
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def init_db(db_path):
    """Initializes the SQLite database and creates tables if they don't exist."""
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create posts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            subreddit TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT,
            author TEXT,
            score INTEGER,
            created_utc REAL,
            url TEXT,
            summary TEXT,
            cluster_id INTEGER
        )
    ''')

    # Create comments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            post_id TEXT NOT NULL,
            author TEXT,
            body TEXT,
            score INTEGER,
            created_utc REAL,
            FOREIGN KEY (post_id) REFERENCES posts (id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

def main():
    """Initializes the Reddit instance, scrapes posts, and stores them in the DB."""
    start_time = time.time()
    total_new_posts = 0
    total_summaries_generated = 0
    total_tokens_used = 0
    
    config = load_config()
    db_path = os.path.join(os.path.dirname(__file__), '..', config['database']['path'])
    
    # Step 1: Initialize the database
    init_db(db_path)
    
    # Step 2: Connect to Reddit API
    reddit_config = config['reddit']
    try:
        reddit = praw.Reddit(
            client_id=reddit_config['client_id'],
            client_secret=reddit_config['client_secret'],
            user_agent=reddit_config['user_agent'],
        )
        print("Successfully connected to Reddit API.")
    except Exception as e:
        print(f"Failed to connect to Reddit: {e}")
        return

    scraper_config = config.get('scraper', {}) 
    sort_by = scraper_config.get('sort_by', 'hot') # Default to 'hot' if not specified
    limit = scraper_config.get('limit', 50)
    time_filter = scraper_config.get('time_filter', 'day')

    comment_config = scraper_config.get('comments', {})
    comments_enabled = comment_config.get('enabled', True)
    comment_limit = comment_config.get('limit_per_post', 10)
    comment_min_score = comment_config.get('min_score', 5)
    
    print(f"Scraper configured to get {limit} '{sort_by}' posts.")
    print(f"Scraper configured to get {limit} '{comment_limit}' comments.")

    # Step 3: Scrape posts
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    def process_batch(batch_data, cursor):
        """Processes a batch of posts, gets summaries, and saves to DB."""
        nonlocal total_new_posts, total_summaries_generated # Allows modifying outer scope variables
        
        if not batch_data:
            return 0

        # Prepare the data for the AI
        posts_for_ai = [{"id": p_data['post'].id, "text": p_data['text']} for p_data in batch_data]
        
        # Get summaries for the entire batch in a single API call
        summaries_map = summarize_batch(posts_for_ai)
        
        processed_count = 0
        if summaries_map:
            # Loop through the original posts from the batch
            for p_data in batch_data:
                post_obj = p_data['post']
                summary_text = summaries_map.get(post_obj.id)
                
                if summary_text:
                    # Save the post with its summary
                    cursor.execute('''
                        INSERT INTO posts (id, subreddit, title, body, author, score, created_utc, url, summary)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        post_obj.id, post_obj.subreddit.display_name, post_obj.title, post_obj.selftext, 
                        str(post_obj.author), post_obj.score, post_obj.created_utc, post_obj.url, summary_text
                    ))
                    
                    # Save the comments that we stored for this specific post
                    for comment in p_data['comments']:
                        cursor.execute('''
                            INSERT OR IGNORE INTO comments (id, post_id, author, body, score, created_utc)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            comment.id, post_obj.id, str(comment.author), comment.body, 
                            comment.score, comment.created_utc
                        ))
                    
                    total_new_posts += 1
                    total_summaries_generated += 1
                    processed_count += 1
        
        return processed_count
    
    subreddits_to_scrape = config['subreddits']
    print(f"Starting to scrape subreddits: {subreddits_to_scrape}")

    for subreddit_name in subreddits_to_scrape:
        print(f"\n--- Scraping r/{subreddit_name} ---")
        subreddit = reddit.subreddit(subreddit_name)
        
        batch_data_to_process = []
        BATCH_SIZE = scraper_config.get('batch_size', 10)

        posts_to_scrape = None
        if sort_by == 'hot':
            posts_to_scrape = subreddit.hot(limit=limit)
        elif sort_by == 'new':
            posts_to_scrape = subreddit.new(limit=limit)
        elif sort_by == 'top':
            print(f"Using time filter: '{time_filter}'")
            posts_to_scrape = subreddit.top(time_filter=time_filter, limit=limit)
        else:
            print(f"Error: Unknown sort_by value '{sort_by}'. Defaulting to hot.")
            posts_to_scrape = subreddit.hot(limit=limit)

        new_post_count = 0
        for post in posts_to_scrape:
            # Check if post already exists. If it does, we skip it entirely.
            cursor.execute("SELECT id FROM posts WHERE id = ?", (post.id,))
            if cursor.fetchone() is not None:
                continue
            
            # --- THIS IS THE CORRECT LOGICAL FLOW FOR A NEW POST ---
            print(f"Found new post, adding to batch: '{post.title[:60]}...'")
            
            # 1. Scrape comments and build text (same as before)
            post.comments.replace_more(limit=0)
            top_comments = post.comments.list()[:comment_limit]
            
            discussion_text = f"Post Title: {post.title}\nPost Body: {post.selftext}\n\n--- Comments ---\n"
            for i, comment in enumerate(top_comments):
                if not isinstance(comment, praw.models.MoreComments) and comment.body:
                    discussion_text += f"Comment {i+1}: {comment.body}\n"
            
            # 2. Store everything we need for this post in one dictionary
            batch_data_to_process.append({
                "post": post,
                "comments": top_comments,
                "text": discussion_text
            })

            # 3. If the batch is full, process it using our helper function
            if len(batch_data_to_process) >= BATCH_SIZE:
                count = process_batch(batch_data_to_process, cursor)
                posts_processed_this_subreddit += count
                print(f"Processed a batch of {count} posts.")
                batch_data_to_process = [] # Clear the batch

        # After the loop, process any remaining posts in the last batch
        if batch_data_to_process:
            count = process_batch(batch_data_to_process, cursor)
            posts_processed_this_subreddit += count
            print(f"Processed the final batch of {count} posts.")
            

        conn.commit() # Commit all changes for this subreddit at once
        print(f"Finished r/{subreddit_name}. Stored {posts_processed_this_subreddit} new summarized posts.")

    conn.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n--- SCRAPING & ANALYSIS COMPLETE ---")
    print(f"      Total Time Elapsed: {elapsed_time:.2f} seconds")
    print(f"        Total New Posts Found: {total_new_posts}")
    print(f"Total Summaries Generated: {total_summaries_generated}")
    print(f"         Total Tokens Used: {total_tokens_used}")
    print("------------------------------------")

if __name__ == "__main__":
    main()