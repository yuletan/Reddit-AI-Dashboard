
'''from ai.summarize import summarize_text'''
from ai.summarize import summarize_batch
import praw
import yaml
import sqlite3
import os
import time 
import re

def load_config():
    """Loads the configuration from settings.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def init_db(db_path):
    """Initializes the SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY, subreddit TEXT NOT NULL, title TEXT NOT NULL,
            body TEXT, author TEXT, score INTEGER, created_utc REAL, url TEXT,
            summary TEXT, cluster_id INTEGER, sentiment REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY, post_id TEXT NOT NULL, author TEXT, body TEXT,
            score INTEGER, created_utc REAL, FOREIGN KEY (post_id) REFERENCES posts (id)
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

# In scraper/scrape.py, replace the old clean_summary function

def clean_summary(summary_text: str) -> str:
    """
    Aggressively cleans the AI's output to find and return only the
    final summary paragraph.
    """
    if not summary_text or summary_text == "NoSummaryGenerated":
        return "NoSummaryGenerated"
    

    # 1. Remove any "thinking" blocks or XML-style tags
    text = re.sub(r'<.*?>|\[.*?\]|<｜.*?｜>', '', summary_text, flags=re.DOTALL)
    
    # 2. Find the core summary by removing common conversational preambles.
    # This looks for phrases like "Here is the summary:" and takes everything AFTER them.
    preamble_patterns = [
        r'Here is the summary you requested:',
        r'Here is the summary paragraph:',
        r'Here is the summary:',
        r'Summary:',
        r'The following is a summary:'
    ]
    for pattern in preamble_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE)[1]
            break # Stop after the first match

    # 3. Basic cleaning from before
    text = re.sub(r'^\s*(Comments|Post Title):?\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^\s*[\*\-]\s*|\d+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. Final quality check
    if len(text.split()) < 5:
        return "NoSummaryGenerated"
        
    return text

def main():
    """Initializes Reddit, scrapes posts, summarizes them one-by-one, and stores them."""
    start_time = time.time()
    total_new_posts = 0
    
    config = load_config()
    db_path = os.path.join(os.path.dirname(__file__), '..', config['database']['path'])
    init_db(db_path)
    
    try:
        reddit = praw.Reddit(
            client_id=config['reddit']['client_id'],
            client_secret=config['reddit']['client_secret'],
            user_agent=config['reddit']['user_agent'],
        )
        print("Successfully connected to Reddit API.")
    except Exception as e:
        print(f"Failed to connect to Reddit: {e}")
        return

    scraper_config = config.get('scraper', {})
    limit = scraper_config.get('limit', 30) # A smaller limit is better for one-by-one

    batch_size = scraper_config.get('batch_size', 5)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    subreddits_to_scrape = config.get('subreddits', [])
    print(f"Starting to scrape subreddits: {subreddits_to_scrape}")

    for subreddit_name in subreddits_to_scrape:
        print(f"\n--- Scraping r/{subreddit_name} ---")
        subreddit = reddit.subreddit(subreddit_name)
        posts_to_scrape = subreddit.hot(limit=limit)
        posts_processed_this_subreddit = 0
        post_batch = []
        for post in posts_to_scrape:
            # 1. Check if post already exists
            cursor.execute("SELECT id FROM posts WHERE id = ?", (post.id,))
            if cursor.fetchone() is not None:
                continue

            print(f"Found new post: '{post.title[:50]}...'")
            
            # 3. Build the text for summarization
            post.comments.replace_more(limit=0)
            top_comments = post.comments.list()[:scraper_config.get('comments', {}).get('limit_per_post', 10)]
            discussion_text = f"Post Title: {post.title}\nPost Body: {post.selftext}\n\n--- Comments ---\n"
            for comment in top_comments:
                if hasattr(comment, 'body'):
                    discussion_text += f"Comment: {comment.body}\n"

            if len(discussion_text) < 200:
                print(f"  -> Skipping post with insufficient discussion text: '{post.title[:50]}...'")
                continue
            
            max_chars = 20000
            if len(discussion_text) > max_chars:
                print(f"    -> Truncating text from {len(discussion_text)} to {max_chars} characters.")
                discussion_text = discussion_text[:max_chars]

            post_data = {
                "id": post.id,
                "text": discussion_text,
                "post_obj": post,
                "top_comments": top_comments
            }
            post_batch.append(post_data)
            print(f"  -> Added to batch. Batch size is now {len(post_batch)}/{batch_size}.")

            # 5. When the batch is full, send it to our new Gemini batch function.
            if len(post_batch) >= batch_size:
                print(f"\n--- Batch full. Processing {len(post_batch)} posts with Gemini... ---")
                
                summaries_map = summarize_batch([{"id": p["id"], "text": p["text"]} for p in post_batch])

                if summaries_map:
                    for p_data in post_batch:
                        post_obj = p_data["post_obj"]
                        # Get the summary from the map returned by the API
                        raw_summary = summaries_map.get(post_obj.id, "NoSummaryGenerated")
                        print(f"    -> Raw AI Summary for {post_obj.id}: '{raw_summary}'")
                        summary_text = clean_summary(raw_summary)

                        if summary_text != "NoSummaryGenerated":
                            print(f"    -> Saving summary for post: {post_obj.id}")
                            # (The database insertion code is the same as your original)
                            cursor.execute('''
                                INSERT INTO posts (id, subreddit, title, body, author, score, created_utc, url, summary)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                post_obj.id, post_obj.subreddit.display_name, post_obj.title, post_obj.selftext,
                                str(post_obj.author), post_obj.score, post_obj.created_utc, post_obj.url, summary_text
                            ))

                            for comment in p_data["top_comments"]:
                                if hasattr(comment, 'id'):
                                    cursor.execute('INSERT OR IGNORE INTO comments (id, post_id, author, body, score, created_utc) VALUES (?, ?, ?, ?, ?, ?)',
                                                   (comment.id, post_obj.id, str(comment.author), comment.body, comment.score, comment.created_utc))
                            
                            posts_processed_this_subreddit += 1
                            total_new_posts += 1
                        else:
                            print(f"    -> AI failed to generate summary for post {post_obj.id}. Skipping.")
                
                # IMPORTANT: Clear the batch to start collecting the next one!
                post_batch = []
                print("--- Batch processing complete. Resuming scraping. ---\n")

        if post_batch: # Check for any remaining posts that didn't fill a full batch
            print(f"\n--- Processing final batch of {len(post_batch)} leftover posts... ---")
            summaries_map = summarize_batch([{"id": p["id"], "text": p["text"]} for p in post_batch])

            if summaries_map:
                for p_data in post_batch:
                    post_obj = p_data["post_obj"]
                    raw_summary = summaries_map.get(post_obj.id, "NoSummaryGenerated")
                    print(f"    -> Raw AI Summary for {post_obj.id}: '{raw_summary}'")
                    summary_text = clean_summary(raw_summary)
                    if summary_text != "NoSummaryGenerated":
                        print(f"    -> Saving summary for post: {post_obj.id}")
                        # Using the full, correct insertion code here
                        cursor.execute('''
                            INSERT INTO posts (id, subreddit, title, body, author, score, created_utc, url, summary)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            post_obj.id, post_obj.subreddit.display_name, post_obj.title, post_obj.selftext,
                            str(post_obj.author), post_obj.score, post_obj.created_utc, post_obj.url, summary_text
                        ))

                        for comment in p_data["top_comments"]:
                            if hasattr(comment, 'id'):
                                cursor.execute('''
                                    INSERT OR IGNORE INTO comments (id, post_id, author, body, score, created_utc) 
                                    VALUES (?, ?, ?, ?, ?, ?)
                                ''', (comment.id, post_obj.id, str(comment.author), comment.body, comment.score, comment.created_utc))
                        posts_processed_this_subreddit += 1
                        total_new_posts += 1

            # Commit all changes for this subreddit
            conn.commit()
            print(f"Finished r/{subreddit_name}. Stored {posts_processed_this_subreddit} new summarized posts.")

            # 4. Summarize this single post
            '''raw_summary = summarize_text(discussion_text)
            print(f"DEBUG: Raw AI Response: '{raw_summary}'")
            
            # 5. Clean the summary
            summary_text = clean_summary(raw_summary)
            
            
            # 6. Save to database if the summary is valid
            if summary_text != "NoSummaryGenerated":
                print(f"    -> AI Summary: {summary_text[:100]}...")
                cursor.execute('
                    INSERT INTO posts (id, subreddit, title, body, author, score, created_utc, url, summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ', (
                    post.id, post.subreddit.display_name, post.title, post.selftext, 
                    str(post.author), post.score, post.created_utc, post.url, summary_text
                ))
                
                # Save comments as before
                for comment in top_comments:
                    if hasattr(comment, 'id'):
                        cursor.execute('INSERT OR IGNORE INTO comments (id, post_id, author, body, score, created_utc) VALUES (?, ?, ?, ?, ?, ?)',
                                       (comment.id, post.id, str(comment.author), comment.body, comment.score, comment.created_utc))
                
                conn.commit()
                posts_processed_this_subreddit += 1
                total_new_posts += 1
            
            
            else:
                print("    -> AI failed to generate a valid summary. Skipping post.")'''



    conn.close()
    end_time = time.time()
    print("\n--- SCRAPING & ANALYSIS COMPLETE ---")
    print(f"      Total Time Elapsed: {end_time - start_time:.2f} seconds")
    print(f"        Total New Posts Found: {total_new_posts}")
    print("------------------------------------")

if __name__ == "__main__":
    main()