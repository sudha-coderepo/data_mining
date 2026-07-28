import os
import sys
import time
import pandas as pd
import numpy as np

# Load local environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Resilient SDK import
try:
    from google import genai
    from google.genai import types
    SDK_VERSION = "new"
except ImportError:
    try:
        import google.generativeai as genai
        SDK_VERSION = "legacy"
    except ImportError:
        print("Error: Missing Gemini API libraries.")
        print("Please install the Gemini SDK using: pip install google-genai")
        sys.exit(1)

try:
    from sklearn.metrics import accuracy_score, cohen_kappa_score
except ImportError:
    print("Warning: scikit-learn is not installed. Metrics calculation will run in fallback mode.")

def initialize_client():
    use_vertex = os.environ.get("VERTEX_AI", "false").lower() == "true"
    project_id = os.environ.get("GCP_PROJECT")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    
    if use_vertex:
        print(f"Initializing Client using Google Cloud Vertex AI (Project: {project_id}, Location: {location})...")
        if SDK_VERSION == "new":
            return genai.Client(vertexai=True, project=project_id, location=location)
        else:
            print("Error: Vertex AI integration requires the modern 'google-genai' SDK.")
            print("Please upgrade by running: pip install -U google-genai")
            sys.exit(1)
            
    # Fallback to API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[!] Error: GEMINI_API_KEY environment variable not found.")
        print("Please set your API key in your terminal before running:")
        print("  On Windows PowerShell: $env:GEMINI_API_KEY=\"your_api_key_here\"")
        print("  On Windows CMD:        set GEMINI_API_KEY=your_api_key_here")
        print("  On Linux/macOS:        export GEMINI_API_KEY=\"your_api_key_here\"\n")
        sys.exit(1)
        
    if SDK_VERSION == "new":
        return genai.Client(api_key=api_key)
    else:
        genai.configure(api_key=api_key)
        return genai

# Fallback models ordered by priority (supported on Google Cloud Vertex AI)
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash"
]

def call_gemini(client, text):
    prompt = f"""
You are an expert customer feedback analyzer. Analyze the following product review and extract:
1. Sentiment: Must be exactly one of: "Positive", "Neutral", "Negative".
2. Category: Suggest the best matching complaint or request theme. It must be exactly one of:
   - "Delivery issue" (shipping delay, broken arrival, tracking problems)
   - "Product quality issue" (defects, durability, poor material, performance)
   - "Price complaint" (too expensive, value-for-money complaints)
   - "Customer service issue" (difficulty returning, refund issues, rude support)
   - "Feature request" (suggestions, missing features, request for enhancements)
   - "Other" (general reviews, positive feedback with no complaints)

Return ONLY a valid JSON object matching this schema. Do not include markdown code block formatting (no backticks):
{{
  "sentiment": "Positive/Neutral/Negative",
  "category": "category_name"
}}

Review to analyze:
"{text}"
"""
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            if SDK_VERSION == "new":
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                return response.text
            else:
                # Legacy SDK fallback
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                return response.text
        except Exception as e:
            last_error = e
            error_msg = str(e)
            print(f"Model {model_name} failed: {error_msg[:120]}...")
            
            # Check if it is a quota or availability error to determine if we should fallback
            is_quota_error = "429" in error_msg or "Quota" in error_msg or "rate limit" in error_msg.lower()
            is_not_found = "404" in error_msg or "not found" in error_msg.lower()
            
            if is_quota_error or is_not_found:
                print(f"Skipping model {model_name} due to rate-limit/not-found. Trying fallback...")
                continue
            else:
                # If it's a different error (e.g. bad API key format), break early
                break
                
    print(f"All models failed. Last error: {last_error}")
    return None


def main():
    cleaned_file = "cleaned_amazon_reviews.csv"
    if not os.path.exists(cleaned_file):
        print(f"Error: Could not find '{cleaned_file}'. Did you complete Phase 1?")
        sys.exit(1)
        
    df = pd.read_csv(cleaned_file)
    num_to_label = len(df)
    
    print(f"Loaded full dataset: {num_to_label} clean reviews.")
    print("WARNING: Running on the full dataset with the Gemini Free Tier will take approximately 80 minutes")
    print("due to the 15 Requests-Per-Minute (RPM) rate limit (requires ~4.1 seconds sleep per review).")
    
    # Initialize the Gemini API client
    client = initialize_client()
    
    llm_sentiments = []
    llm_categories = []
    processed_indices = []
    
    # Check if we have progress saved to resume (in case of disconnection or rate limit blocks)
    progress_file = "reviews_labeled_llm_progress.csv"
    if os.path.exists(progress_file):
        print(f"Found existing progress file: {progress_file}. Resuming...")
        progress_df = pd.read_csv(progress_file)
        processed_indices = list(progress_df.index)
        llm_sentiments = list(progress_df['llm_sentiment'])
        llm_categories = list(progress_df['llm_category'])
        print(f"Resuming from index {len(processed_indices)}...")
    
    start_time = time.time()
    
    print("\nStarting LLM Labeling... (this will call the Gemini API)")
    for i in range(len(processed_indices), num_to_label):
        text = df.iloc[i]['reviews.text']
        
        # Calculate progress and ETA
        completed = i
        remaining = num_to_label - completed
        elapsed = time.time() - start_time
        avg_time = elapsed / completed if completed > 0 else 4.2
        eta_min = (remaining * avg_time) / 60.0
        
        print(f"[{i+1}/{num_to_label}] Processing review. ETA: {eta_min:.1f} minutes. Text: {text[:45]}...")
        
        # Call Gemini API
        result_json_str = call_gemini(client, text)
        
        if result_json_str:
            import json
            try:
                # Clean up any potential markdown wraps from model response
                clean_str = result_json_str.strip().replace("```json", "").replace("```", "")
                result_data = json.loads(clean_str)
                
                sentiment = result_data.get("sentiment", "Neutral")
                category = result_data.get("category", "Other")
                
                # Standardize casing
                if sentiment.lower() == "positive": sentiment = "Positive"
                elif sentiment.lower() == "negative": sentiment = "Negative"
                else: sentiment = "Neutral"
                
                llm_sentiments.append(sentiment)
                llm_categories.append(category)
                processed_indices.append(i)
                
                # Periodically save progress every 5 reviews in case the script is interrupted
                if len(processed_indices) % 5 == 0:
                    temp_df = df.iloc[processed_indices].copy()
                    temp_df['llm_sentiment'] = llm_sentiments
                    temp_df['llm_category'] = llm_categories
                    temp_df.to_csv(progress_file, index=False)
                    
            except Exception as e:
                print(f"Error parsing JSON output for review {i+1}: {e}. Response was: {result_json_str}")
                # Append fallback values so indexing stays aligned
                llm_sentiments.append("Neutral")
                llm_categories.append("Other")
                processed_indices.append(i)
        else:
            print(f"Skipping review {i+1} due to API error. Inserting fallback values.")
            llm_sentiments.append("Neutral")
            llm_categories.append("Other")
            processed_indices.append(i)
            
        # Enforce rate limits based on subscription tier
        use_vertex = os.environ.get("VERTEX_AI", "false").lower() == "true"
        if use_vertex:
            time.sleep(0.05)  # Vertex AI paid tier supports high speed requests
        else:
            time.sleep(4.2)   # Free tier requires a 4.2s delay to avoid 429 errors
            
    # Subset of dataframe that was labeled
    labeled_df = df.iloc[processed_indices].copy()
    labeled_df['llm_sentiment'] = llm_sentiments
    labeled_df['llm_category'] = llm_categories
    
    # Save the final labeled output
    output_filename = "reviews_labeled_llm.csv"
    labeled_df.to_csv(output_filename, index=False)
    print(f"\nLabeling completed! Output saved to: {os.path.abspath(output_filename)}")
    
    # Remove progress file after successful completion
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    # Evaluate agreement
    y_true = labeled_df['sentiment'].values
    y_pred = labeled_df['llm_sentiment'].values
    
    print("\n--- AI-Generated vs Human (Rating-based) Sentiment Alignment ---")
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Sentiment Agreement Accuracy: {acc:.2%}")
    print(f"Cohen's Kappa Score: {kappa:.4f}")
    
    print("\nSuggested LLM Category Counts:")
    print(labeled_df['llm_category'].value_counts())

if __name__ == "__main__":
    main()
