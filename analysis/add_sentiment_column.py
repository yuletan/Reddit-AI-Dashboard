# add_sentiment_column.py
import sqlite3
import yaml
import os

def load_config():
    """Loads the configuration from settings.yaml."""
    # os.path.dirname(__file__) gets the current folder ('analysis').
    # '..' tells it to go UP one level to the project root.
    # 'config' and 'settings.yaml' tell it to go DOWN into the correct folder and file.
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    print("--- Attempting to add 'sentiment' column to the 'posts' table ---")
    config = load_config()
    db_path = config['database']['path'] # The path from settings.yaml, e.g., "data/reddit_data.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE posts ADD COLUMN sentiment REAL")
        conn.commit()
        conn.close()
        print("✅ Successfully added the 'sentiment' column.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ Column 'sentiment' already exists. No changes needed.")
        else:
            print(f"❌ An unexpected database error occurred: {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()