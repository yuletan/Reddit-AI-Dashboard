import openai
import yaml
import os
import tiktoken 
import json
import time

def load_config():
    """Loads the configuration from settings.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def summarize_text(text_to_summarize):
    """
    Summarizes a given text using the specified OpenRouter model.
    """
    config = load_config()
    ai_config = config.get('ai', {})
    model_to_use = ai_config.get('model', 'deepseek/deepseek-chat-v3.1:free')
    
    client = openai.OpenAI(
        base_url=ai_config.get('api_base'),
        api_key=ai_config.get('api_key'),
    )

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text_to_summarize))
    except Exception:
        # Fallback if tiktoken fails for some reason
        num_tokens = len(text_to_summarize) // 4 

    system_prompt = "You are an expert assistant. Summarize the following Reddit discussion in a single, concise paragraph. Focus on the main topic, key questions, and the most helpful answers or conclusions."

    try:
        print(f"  -> Contacting model: {model_to_use}...")
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_summarize},
            ],
            temperature=0.7,
            max_tokens=150,
        )
        summary = response.choices[0].message.content.strip()
        return summary, num_tokens
    except Exception as e:
        print(f"An error occurred while summarizing: {e}")
        return None, 0
    


def summarize_batch(posts_to_summarize: list):
    """
    Summarizes a batch of posts in a single API call.
    'posts_to_summarize' should be a list of dictionaries, 
    each with an 'id' and 'text' key.
    """
    config = load_config()
    ai_config = config.get('ai', {})
    model_to_use = ai_config.get('model', 'deepseek/deepseek-chat-v3.1:free')
    client = openai.OpenAI(
    base_url=ai_config.get('api_base'),
    api_key=ai_config.get('api_key'),
    )

    # 1. Format the input for the AI
    input_data = {
        "discussions": posts_to_summarize
    }
    input_json = json.dumps(input_data)

    # 2. The new system prompt for batching
    system_prompt = """
    You are an efficient batch processing AI. You will be given a JSON object 
    containing a list of Reddit discussions, each with a unique ID. 
    Your task is to summarize EACH discussion individually. 
    Respond with a single, valid JSON object that maps each original ID 
    to its corresponding summary. Do not include any text outside of the JSON response.
    """

    try:
        print(f"  -> Sending batch of {len(posts_to_summarize)} posts to AI...")
        response = client.chat.completions.create(
            model=model_to_use,
            response_format={"type": "json_object"}, # Ask for JSON output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_json},
            ],
            temperature=0.5,
        )
        
        # 3. Parse the JSON response from the AI
        response_content = response.choices[0].message.content
        summaries_map = json.loads(response_content)
        return summaries_map

    except json.JSONDecodeError:
        print("  -> ERROR: AI did not return valid JSON. Batch failed.")
        return None
    except Exception as e:
        print(f"  -> An error occurred during batch summarization: {e}")
        return None
    
def fake_summarize_batch(posts_to_summarize: list):
    """
    A mock function that simulates the AI summarizer without making an API call.
    """
    print("  -> Using FAKE summarizer. NO API CALLS WILL BE MADE.")
    time.sleep(1) 
    
    fake_summaries_map = {}
    for post in posts_to_summarize:
        post_id = post.get('id')
        fake_summaries_map[post_id] = f"This is a fake summary for post ID {post_id}."
        
    return fake_summaries_map


if __name__ == '__main__':
    print("--- Testing BATCH summarization module ---")
    
    # Create a sample batch of posts, just like the scraper would
    sample_batch = [
        {"id": "post_1", "text": "Post about Python decorators and how they work."},
        {"id": "post_2", "text": "A user is asking for career advice for data science jobs."},
        {"id": "post_3", "text": "Help, I have a KeyError in my pandas DataFrame!"}
    ]
    
    print(f"\nTesting with a batch of {len(sample_batch)} posts.")
    
    # We will test the FAKE function, since you have no API calls left
    summaries = fake_summarize_batch(sample_batch)
    
    if summaries:
        print("\n--- Batch Summary Results ---")
        # Pretty print the JSON-like dictionary
        print(json.dumps(summaries, indent=2))
    else:
        print("\nBatch summarization failed.")