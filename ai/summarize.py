import openai
import google.generativeai as genai
import yaml
import os
import tiktoken 
import json
import time
import requests

def load_config():
    """Loads the configuration from settings.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


'''def summarize_text(text_to_summarize: str):
    """
    Summarizes a single given text using the Google Gemini API.
    """
    config = load_config()
    gemini_config = config.get('gemini', {})
    
    api_key = gemini_config.get('api_key')
    if not api_key or "YOUR_GEMINI_API_KEY_HERE" in api_key:
        print("--- ERROR: Gemini API key not found or not set in config/settings.yaml ---")
        return None

    try:
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 250,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]

        # --- THIS IS THE MODIFIED SECTION ---
        # Read the model name from the config, with a fallback default.
        model_to_use = gemini_config.get('model_name', 'gemini-2.5-flash-lite-preview-09-2025')

        model = genai.GenerativeModel(
            model_name=model_to_use, # Use the variable here
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        # --- END OF MODIFICATION ---
        
        system_prompt = system_prompt = """
You are a helpful Reddit user who is skilled at summarizing long discussions. Your task is to read the following Reddit post and its comments and then write a new, single paragraph that explains the conversation to someone who hasn't read the thread.

**CRITICAL FORMATTING RULE:**
Your entire response MUST be a single, continuous paragraph of natural-sounding text. It must not contain any markdown headers (like '##'), bullet points (* or -), numbered lists, or multiple paragraphs.

**STYLE GUIDE:**
- Write in a natural, narrative style.
- Synthesize the main questions and the most helpful answers into a cohesive explanation.
- **NEVER** mention "the post," "the comments," "Comment 1," or "the user." Instead, refer to the content directly (e.g., "The main question was about...", "Several people suggested that...").

**Failure Condition:** If the text cannot be summarized into a helpful narrative paragraph, your entire output must be the single phrase: `NoSummaryGenerated`.
"""
        
        prompt_parts = [system_prompt, "\n---\n", text_to_summarize]

        print(f"  -> Contacting Google Gemini API ({model_to_use})...") # Updated print statement
        
        response = model.generate_content(prompt_parts)
        
        summary = response.text.strip()
        
        time.sleep(25) 
        
        return summary

    except Exception as e:
        print(f"--- An error occurred while contacting the Gemini API: {e} ---")
        return None
'''

'''def summarize_text(text_to_summarize: str):
    """
    Summarizes a single given text using the specified OpenRouter model.
    """
    config = load_config()
    ai_config = config.get('ai', {})
    
    # Check if API key is present
    api_key = ai_config.get('api_key')
    if not api_key or "YOUR_OPENROUTER_API_KEY_HERE" in api_key:
        print("--- ERROR: OpenRouter API key not found or not set in config/settings.yaml ---")
        return None

    client = openai.OpenAI(
        base_url=ai_config.get('api_base'),
        api_key=api_key,
    )

    # Use the high-quality "Helpful Redditor" persona prompt
    system_prompt = """
    You are a helpful Reddit user who is skilled at summarizing long discussions. Your task is to read the following Reddit post and its comments and then write a new, single paragraph that explains the conversation to someone who hasn't read the thread.

    **Your summary MUST:**
    - Be written in a natural, narrative style.
    - Flow like a normal paragraph, not a list or a report.
    - Synthesize the main questions and the most helpful answers into a cohesive explanation.
    - **NEVER** mention "the post," "the comments," "Comment 1," or "the user." Instead, refer to the content directly (e.g., "The main question was about...", "Several people suggested that...").
    - Be ready for direct display on a dashboard without any extra text or labels.

    **Failure Condition:** If the text cannot be summarized into a helpful narrative, your entire output must be the single phrase: `NoSummaryGenerated`.

    Begin summary now.
    """

    try:
        model_to_use = ai_config.get('model', 'deepseek/deepseek-chat-v3.1:free')
        print(f"  -> Contacting model: {model_to_use}...")
        
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_summarize},
            ],
            temperature=0.7,
            max_tokens=250,
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Add a small delay to respect API rate limits
        time.sleep(1) # 1-second delay between calls
        
        return summary

    except Exception as e:
        print(f"--- An error occurred while contacting OpenRouter: {e} ---")
        return None
'''

'''def summarize_batch(posts_to_summarize: list):
    """
    Summarizes a batch of posts by sending them to our custom Colab-hosted AI model.
    'posts_to_summarize' should be a list of dictionaries, 
    each with an 'id' and 'text' key.
    """
    
    # !!! IMPORTANT !!!
    # Paste the ngrok URL you copied from your Google Colab notebook here
    # It must include the "/summarize" endpoint at the end.
    colab_url = "https://unsimmered-unstout-kaydence.ngrok-free.dev/summarize"

    # 1. Format the input data
    input_data = {
        "discussions": posts_to_summarize
    }
    
    # Convert the dictionary to a JSON string
    input_json = json.dumps(input_data, indent=2)

    try:
        print(f"  -> Sending batch of {len(posts_to_summarize)} posts to Colab AI...")
        
        # 2. Make the HTTP POST request to our server
        response = requests.post(
            colab_url, 
            headers={
                "Content-Type": "application/json",
                # ngrok's free tier sometimes shows an interstitial page. This header helps bypass it.
                "ngrok-skip-browser-warning": "true" 
            },
            data=input_json,
            timeout=400  # Give it up to 5 minutes to process a large batch
        )
        
        # Raise an exception if the request was not successful (e.g., 404 Not Found, 500 Internal Server Error)
        response.raise_for_status()
        
        # 3. Parse the JSON response from the server
        summaries_map = response.json()
        return summaries_map

    except requests.exceptions.RequestException as e:
        print(f"\n--- ERROR: Could not connect to the Colab AI server ---")
        print(f"--- Details: {e} ---")
        print("--- Is the Colab notebook still running and is the ngrok URL correct? ---")
        return None
    except json.JSONDecodeError:
        print("\n--- ERROR: The server's response was not valid JSON ---")
        print("--- RESPONSE TEXT FROM SERVER ---")
        print(response.text)
        print("---------------------------------\n")
        return None
    except Exception as e:
        print(f"\n--- An unexpected error occurred during batch summarization: {e} ---")
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
        
    return fake_summaries_map'''

def summarize_batch(posts_to_summarize: list):
    """
    Summarizes a batch of posts by combining them into a single prompt,
    making ONE API call to Gemini, and parsing the structured JSON response.
    'posts_to_summarize' should be a list of dictionaries, each with an 'id' and 'text' key.
    """
    print(f"--- Starting RELIABLE SEQUENTIAL batch for {len(posts_to_summarize)} posts ---")
    
    # 1. Load config and initialize the model (same as before)
    config = load_config()
    gemini_config = config.get('gemini', {})
    api_key = gemini_config.get('api_key')

    if not api_key or "YOUR_GEMINI_API_KEY_HERE" in api_key:
        print("--- ERROR: Gemini API key not found. ---")
        return None
    
    try:
        genai.configure(api_key=api_key)
        # We might need more output tokens for a combined response
        generation_config = {"temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 20000}

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]

        model_to_use = gemini_config.get('model_name', 'gemini-2.5-flash-lite')
        model = genai.GenerativeModel(model_name=model_to_use, generation_config=generation_config, safety_settings=safety_settings)

        # 2. --- CRITICAL --- Create the new System Prompt for JSON batch processing
        system_prompt = system_prompt = """
You are an expert AI discussion analyst. Your primary skill is to read a conversation and distill its essence into a compelling, insightful narrative. You will be given a batch of discussions, each with a unique ID.

Your task is to craft a summary for each discussion that captures not just the topic, but the dynamic of the conversation itself.

**CRITICAL OUTPUT FORMATTING RULE:**
Your entire response MUST be a single, valid JSON object, starting with `{` and ending with `}`. Do not include any text or markdown formatting outside of this JSON structure. The JSON object must map the provided post IDs to their summary strings.

**STYLE & CONTENT GUIDE (for each summary value in the JSON):**

Each summary must be a single, cohesive paragraph that weaves together the answers to these questions:

1.  **The Catalyst:** What was the initial question, problem, or observation that started the discussion?
2.  **The Dominant Narrative:** What was the most common or upvoted opinion, solution, or explanation? What was the core reasoning behind this view?
3.  **The Counter-Narrative:** Was there a significant dissenting opinion or alternative perspective? Explain the reasoning for this counter-point. Highlighting the conflict between viewpoints is crucial for a good summary.
4.  **The Conclusion/Takeaway:** What was the final consensus, or if there wasn't one, what is the key takeaway a reader should have from the debate?

**Go beyond surface-level statements.** Instead of saying "the feedback was mixed," explain *why* it was mixed by presenting the opposing arguments. Extract the 'why' behind the opinions.

**PROHIBITIONS:**
- **NEVER** mention "the post," "the comments," "the user," or "OP."
- Each summary must be a single paragraph.

**Failure Condition:** If a discussion lacks enough substance to create a narrative with conflicting or supporting viewpoints, the value for its key in the JSON must be the single string: `NoSummaryGenerated`.

Example Response Format:
{
  "post_id_1": "When an investor, anxious about a 10% portfolio drop, asked if they should buy the dip, the dominant advice was to proceed if their long-term conviction in the assets remained strong. However, a significant counter-narrative warned against emotionally 'catching a falling knife,' suggesting the investor's anxiety might indicate a risk tolerance unsuitable for their current holdings, leaving the key takeaway that strategy should depend on pre-existing conviction, not short-term fear.",
  "post_id_2": "This is another detailed summary that explains the initial problem, the main solution offered, and a dissenting view.",
  "post_id_3": "NoSummaryGenerated"
}
"""
    except Exception as e:
        print(f"--- Failed to initialize Gemini model: {e} ---")
        return None

    # 3. --- CRITICAL --- Combine all posts into one "mega-prompt"
    mega_prompt = []
    for post in posts_to_summarize:
        post_id = post.get('id')
        text_to_summarize = post.get('text')
        # Clearly separate each post and provide its ID
        mega_prompt.append(f"--- POST START ---\nID: {post_id}\nCONTENT:\n{text_to_summarize}\n--- POST END ---")
    
    # Join them all into one giant string
    final_prompt_text = "\n\n".join(mega_prompt)

    # 4. Make the SINGLE API call
    try:
        print(f"  -> Sending {len(posts_to_summarize)} posts in one API call...")
        prompt_parts = [system_prompt, "\n---\n", final_prompt_text]
        response = model.generate_content(prompt_parts)
        time.sleep(5)
        raw_response_text = response.text.strip()
        print(f"  -> Raw AI Response: '{raw_response_text}'")

        # 5. --- CRITICAL --- Parse the JSON response
        # Find the first '{' and the last '}' to clean up potential conversational padding
        json_start = raw_response_text.find('{')
        json_end = raw_response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            print("--- ERROR: No JSON object found in the AI response. Batch failed. ---")
            return None

        json_string = raw_response_text[json_start:json_end]
        
        summaries_map = json.loads(json_string)
        print("--- Successfully parsed JSON response. Batch complete. ---")

        return summaries_map
         

    except json.JSONDecodeError as e:
        print(f"--- ERROR: Failed to decode JSON from AI response. The format was invalid: {e} ---")
        print("--- The entire batch of summaries will be skipped. ---")
        return None # Return None to indicate the whole batch failed
    except Exception as e:
        print(f"--- An error occurred during the Gemini API call: {e} ---")
        return None


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
    summaries = summarize_batch(sample_batch)
    
    if summaries:
        print("\n--- Batch Summary Results ---")
        # Pretty print the JSON-like dictionary
        print(json.dumps(summaries, indent=2))
    else:
        print("\nBatch summarization failed.")