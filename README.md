# Reddit AI Dashboard

This project is a complete pipeline that automatically scrapes hot discussions from specified subreddits, uses the Google Gemini API to generate concise summaries, groups them into thematic clusters using machine learning, and presents the results in an interactive web dashboard.

## Key Features

-   **Automated Reddit Scraper**: Fetches top posts and their most relevant comments from any subreddit.
-   **Efficient AI Summarization**: Uses a batched approach to summarize dozens of discussions with a single, efficient API call to Google Gemini.
-   **Thematic Clustering**: Applies TF-IDF vectorization and K-Means clustering to automatically group similar discussions into distinct topics.
-   **Incremental & Full Reclustering**: Can either cluster only new posts for speed or perform a full reclustering of all data to improve topic accuracy over time.
-   **Interactive Dashboard**: A web-based UI built with Streamlit that allows you to filter, search, and explore the summarized and clustered Reddit data.

---

### Video Link:
[Reddit AI Dashboard Video](https://youtu.be/IdnuqCl5Z-k?si=f0f6248VrEb2NcAw)

## How It Works: The Data Pipeline

The project follows a clear, step-by-step data processing pipeline:

1.  **Scrape (`scraper/scrape.py`)**:
    -   Connects to the Reddit API using credentials from `config/settings.yaml`.
    -   Fetches a list of hot posts from the target subreddits.
    -   Concurrently fetches the top comments for each new post to maximize speed.
    -   Organizes posts into batches.

2.  **Summarize (`ai/summarize.py`)**:
    -   For each batch, a single, structured API call is sent to the Google Gemini API.
    -   The AI returns a JSON object containing a summary for each post in the batch.

3.  **Store (`data/reddit_data.db`)**:
    -   The scraped post information, comments, and AI-generated summaries are saved into a local SQLite database.

4.  **Cluster (`analysis/cluster.py`)**:
    -   Reads all summarized posts from the database.
    -   Converts the text summaries into numerical vectors (TF-IDF).
    -   Uses the Scikit-learn library to perform K-Means clustering.
    -   Updates the database, assigning a `cluster_id` to each post.

5.  **Visualize (`dashboard/app.py`)**:
    -   Loads the final, processed data from the SQLite database.
    -   Launches a Streamlit web application.
    -   Provides filters and a clean interface to explore the insights.

---

## Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

-   Python 3.8 or newer
-   Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2: Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

-   **On macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
-   **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

### Step 3: Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 4: Configure Your Settings

You must provide your own API keys and settings for the script to work.

1.  Create and navigate the `config` directory.
2.  **Create** the file `settings.yaml`.
3.  Open `settings.yaml` with a text editor and fill in your credentials. See the "Configuration" section below for details.

---

## How to Run the Project

Run the scripts in the following order. Make sure your virtual environment is activated.

### Step 1: Scrape and Summarize Data

This command will fetch new posts from the subreddits you defined in your config, get AI summaries, and save them to the database.

```bash
python scraper/scrape.py
```

### Step 2: Cluster the Summaries

After scraping, run the clustering script. You have two options:

-   **A) Incremental Clustering (Faster, for daily use):**
    This will only cluster the new posts you just scraped.
    ```bash
    python analysis/cluster.py
    ```

-   **B) Full Reclustering (Slower, for periodic model updates):**
    This will erase all old cluster assignments and re-cluster every post in your database. This is useful after you've collected a lot of new data.
    ```bash
    python analysis/cluster.py --recluster
    ```

### Step 3: Launch the Dashboard

Start the Streamlit web server to view your results.

```bash
streamlit run dashboard/app.py
```

Your web browser should automatically open with the dashboard. If not, open the "Network URL" displayed in your terminal.

---

## Project Structure

```
├── analysis/
│   └── cluster.py         # K-Means clustering script
├── ai/
│   └── summarize.py       # AI summarization logic (Gemini API)
├── config/
│   └── settings.yaml      # Your private configuration and API keys
│   └── settings.yaml.example # Example config file
├── dashboard/
│   └── app.py             # Streamlit dashboard application
├── data/
│   └── reddit_data.db     # SQLite database (created on first run)
├── scraper/
│   └── scrape.py          # Reddit data scraping script
├── README.md              # This file
└── requirements.txt       # Python package dependencies
```

## 1. Configuration Details (`config/settings.yaml`)

-   **`reddit`**: Credentials for the Reddit API. You need to create a "script" app on Reddit's developer portal.
    -   `client_id`: Your Reddit app's client ID.
    -   `client_secret`: Your Reddit app's client secret.
    -   `user_agent`: A descriptive name for your script (e.g., `MyInsightScraper/1.0`).
-   **`gemini`**: Settings for the Google Gemini API.
    -   `api_key`: Your API key from Google AI Studio.
    -   `model_name`: The specific Gemini model to use (e.g., `gemini-1.5-flash-latest`).
-   **`database`**:
    -   `path`: The relative path where the database file will be stored.
-   **`scraper`**:
    -   `limit`: The number of "hot" posts to fetch from each subreddit per run.
    -   `batch_size`: The number of posts to group together for a single AI summary API call.
-   **`ai`**:
    -   `num_clusters`: The number of distinct topics K-Means should try to find in the data.
-   **`subreddits`**: A YAML list of subreddits to scrape (do not include the "r/").

### 2. `requirements.txt`

Create a new file named `requirements.txt` in the root directory of your project. This file tells Python's `pip` installer exactly which packages your project needs.

```text
# Reddit API
praw

# Configuration file handling
PyYAML

# Google Gemini API
google-generativeai

# Data handling and analysis
pandas
scikit-learn

# Web Dashboard
streamlit
plotly

# HTTP requests (useful for other APIs)
requests
```


### 3. `settings.yaml`

Create a new file named `settings.yaml` inside the `config/` directory. This serves as a template so you never accidentally commit your secret keys to Git.

```yaml
# ------------------------------------------------------------------
#                Reddit Insight AI Configuration
#
# Instructions:
# 1. Fill in your API keys and credentials below.
# 2. Rename this file from "settings.yaml.example" to "settings.yaml".
# 3. IMPORTANT: Never commit your "settings.yaml" file to a public
#    git repository. Add it to your .gitignore file.
# ------------------------------------------------------------------

# Reddit API Credentials
# Create a "script" application at: https://www.reddit.com/prefs/apps
reddit:
  client_id: "YOUR_REDDIT_CLIENT_ID_HERE"
  client_secret: "YOUR_REDDIT_CLIENT_SECRET_HERE"
  user_agent: "RedditInsightScraper/1.0 by YourUsername"

# Google Gemini API Key
# Get your key from Google AI Studio: https://aistudio.google.com/app/apikey
gemini:
  api_key: "YOUR_GEMINI_API_KEY_HERE"
  # Model to use. 'gemini-1.5-flash-latest' is fast and cost-effective for this task.
  model_name: "gemini-1.5-flash-latest"

# Database file path
database:
  path: "data/reddit_data.db"

# Scraper Settings
scraper:
  # Number of 'hot' posts to retrieve from each subreddit.
  # Reddit API limit is ~1000 for this endpoint. Start with a smaller number.
  limit: 100
  # How many posts to send to the AI in a single batch API call.
  # Higher values are more efficient but use more memory.
  batch_size: 10
  comments:
    # Number of top comments (by 'best' sort) to include in the summary context.
    limit_per_post: 15

# AI & Clustering Settings
ai:
  # The number of topics the K-Means algorithm should find.
  # A good starting point is between 8 and 15.
  num_clusters: 10

# List of subreddits to scrape.
# Do NOT include the "r/" prefix.
subreddits:
  - datascience
  - machinelearning
  - programming
  - python
```
