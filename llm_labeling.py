import os
import sys
import time
import json
import threading
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        sys.exit(1)
        
    if SDK_VERSION == "new":
        return genai.Client(api_key=api_key)
    else:
        genai.configure(api_key=api_key)
        return genai

# Fallback models ordered by priority (supported on Google Cloud Vertex AI / AI Studio)
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
  "category": "Delivery issue/Product quality issue/Price complaint/Customer service issue/Feature request/Other"
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
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                return response.text
            else:
                # Legacy SDK fallback
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json", "temperature": 0.1}
                )
                return response.text
        except Exception as e:
            last_error = str(e)
            # Log failure and try next model
            continue
            
    # If all models fail, raise the last exception
    raise Exception(f"All models failed. Last error: {last_error}")

# Global variables for thread safety
progress_lock = threading.Lock()
csv_lock = threading.Lock()
progress_file = "reviews_labeled_llm_progress.csv"
output_file = "reviews_labeled_llm.csv"

processed_count = 0
total_to_process = 0

def process_review_worker(client, index, row, columns_order):
    global processed_count
    
    text = row['reviews.text']
    
    # Run API call with retry wrapper
    max_retries = 3
    result_json_str = None
    for attempt in range(max_retries):
        try:
            result_json_str = call_gemini(client, text)
            if result_json_str:
                break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[Row {index+1}] Failed after {max_retries} attempts: {e}")
            else:
                time.sleep(1.0 * (attempt + 1)) # linear backoff

    # Parse response
    llm_sentiment = "Neutral"
    llm_category = "Other"
    
    if result_json_str:
        try:
            # Strip backticks or extra lines if present
            cleaned_json = result_json_str.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            cleaned_json = cleaned_json.strip()
            
            data = json.loads(cleaned_json)
            llm_sentiment = data.get("sentiment", "Neutral")
            llm_category = data.get("category", "Other")
            
            # Normalize categories in case of small typo shifts
            valid_categories = ["Delivery issue", "Product quality issue", "Price complaint", "Customer service issue", "Feature request", "Other"]
            if llm_category not in valid_categories:
                llm_category = "Other"
                
            valid_sentiments = ["Positive", "Neutral", "Negative"]
            if llm_sentiment not in valid_sentiments:
                llm_sentiment = "Neutral"
        except Exception as e:
            # JSON parsing error fallback
            pass

    # Create completed row Series/Dict
    output_row = row.copy()
    output_row['llm_sentiment'] = llm_sentiment
    output_row['llm_category'] = llm_category
    
    # Save directly to progress file in a thread-safe manner
    with csv_lock:
        df_row = pd.DataFrame([output_row])[columns_order]
        df_row.to_csv(progress_file, mode='a', header=False, index=False, encoding='utf-8')

    # Update global progress counter
    with progress_lock:
        processed_count += 1
        pct = (processed_count / total_to_process) * 100
        print(f"\r[{processed_count}/{total_to_process}] Labeled: {pct:.1f}% | Last: {llm_sentiment} - {llm_category}", end="", flush=True)

def main():
    global total_to_process, processed_count
    
    # Load dataset
    input_file = "cleaned_amazon_reviews.csv"
    if not os.path.exists(input_file):
        print(f"Error: Could not find clean dataset '{input_file}'. Did you run Phase 1?")
        sys.exit(1)
        
    df = pd.read_csv(input_file)
    total_reviews = len(df)
    print(f"Loaded dataset: {total_reviews} reviews.")

    # Configure limit (Defaults to all 10,000, can be set in .env using LABEL_LIMIT)
    label_limit = os.environ.get("LABEL_LIMIT")
    if label_limit:
        try:
            label_limit = int(label_limit)
            df = df.iloc[:label_limit].copy()
            total_reviews = len(df)
            print(f"Applying LABEL_LIMIT: Processing first {total_reviews} reviews.")
        except ValueError:
            pass

    client = initialize_client()

    # Determine what has already been processed to support resume functionality
    processed_indices = []
    columns_order = list(df.columns) + ['llm_sentiment', 'llm_category']

    if os.path.exists(progress_file):
        print(f"Found existing progress file: {progress_file}. Resuming...")
        try:
            progress_df = pd.read_csv(progress_file)
            processed_ids = set(progress_df['id'].dropna().astype(str).tolist())
            df_to_process = df[~df['id'].astype(str).isin(processed_ids)].copy()
            processed_count = len(processed_ids)
            print(f"Resuming: {processed_count} already labeled. {len(df_to_process)} remaining.")
        except Exception as e:
            print(f"Error reading progress file ({e}). Starting fresh...")
            df_to_process = df.copy()
            # Initialize progress file with header
            pd.DataFrame(columns=columns_order).to_csv(progress_file, index=False, encoding='utf-8')
    else:
        # Initialize fresh progress file with header
        pd.DataFrame(columns=columns_order).to_csv(progress_file, index=False, encoding='utf-8')
        df_to_process = df.copy()

    total_to_process = len(df_to_process)
    
    if total_to_process == 0:
        print("All reviews are already labeled. Generating final file...")
        # Copy progress directly to final output
        if os.path.exists(progress_file):
            pd.read_csv(progress_file).to_csv(output_file, index=False, encoding='utf-8')
        sys.exit(0)

    print(f"\nStarting Parallel LLM Labeling (20 threads)...")
    
    # We use 20 threads to run the API calls in parallel
    max_workers = 20
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, row in df_to_process.iterrows():
            futures.append(executor.submit(process_review_worker, client, index, row, columns_order))
            
        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                # Worker thread exceptions are handled internally, but caught here as fallback
                pass

    elapsed_time = time.time() - start_time
    print(f"\n\nLabeling completed in {elapsed_time/60.0:.2f} minutes!")

    # Load progress file, verify, and write to final CSV
    print(f"Saving final output to {output_file}...")
    final_df = pd.read_csv(progress_file)
    # Ensure there are no duplicate IDs
    final_df = final_df.drop_duplicates(subset=['id'], keep='last')
    final_df.to_csv(output_file, index=False, encoding='utf-8')

    # Metrics evaluation
    if 'sentiment' in final_df.columns and 'llm_sentiment' in final_df.columns:
        print("\n--- AI-Generated vs Human (Rating-based) Sentiment Alignment ---")
        try:
            acc = accuracy_score(final_df['sentiment'], final_df['llm_sentiment'])
            kappa = cohen_kappa_score(final_df['sentiment'], final_df['llm_sentiment'])
            print(f"Sentiment Agreement Accuracy: {acc:.2%}")
            print(f"Cohen's Kappa Score: {kappa:.4f}")
        except Exception as e:
            print(f"Could not calculate metrics: {e}")

    if 'llm_category' in final_df.columns:
        print("\nSuggested LLM Category Counts:")
        print(final_df['llm_category'].value_counts())

if __name__ == "__main__":
    main()
