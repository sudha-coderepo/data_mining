import os
import pandas as pd

def run_eda():
    csv_file = "Amazon Review Dataset.csv"
    
    print(f"Checking for dataset file: {csv_file}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not find '{csv_file}' in the current directory.")
        
    print("Loading dataset using Pandas (handling multi-line text fields)...")
    # Read the CSV file
    df = pd.read_csv(csv_file, low_memory=False)
    
    print("\n--- Phase 1: Basic Dataset Shape & Schema ---")
    print(f"Total rows loaded: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    print("\nColumns present in the dataset:")
    print(list(df.columns))
    
    # 2. Check for missing values in critical columns
    print("\nChecking for missing values in critical columns:")
    text_nulls = df['reviews.text'].isnull().sum()
    rating_nulls = df['reviews.rating'].isnull().sum()
    print(f"Missing values in 'reviews.text': {text_nulls}")
    print(f"Missing values in 'reviews.rating': {rating_nulls}")
    
    # 3. Clean the dataset: Drop rows with missing text or ratings
    print("\nCleaning data: Removing rows with null reviews or ratings...")
    df_clean = df.dropna(subset=['reviews.text', 'reviews.rating']).copy()
    print(f"Cleaned dataset shape: {df_clean.shape[0]} rows remaining.")
    
    # 4. Map ratings to sentiments (Positive, Neutral, Negative)
    # 4-5 stars: Positive, 3 stars: Neutral, 1-2 stars: Negative
    print("\nMapping numerical ratings (1-5) to sentiments...")
    def map_sentiment(rating):
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"
            
    df_clean['sentiment'] = df_clean['reviews.rating'].apply(map_sentiment)
    
    # 5. Output Class Distribution
    print("\nSentiment Class Distribution:")
    print(df_clean['sentiment'].value_counts())
    print("\nNumerical Rating Distribution:")
    print(df_clean['reviews.rating'].value_counts().sort_index())
    
    # 6. Output Category Distribution
    print("\nTop 5 Product Categories:")
    print(df_clean['categories'].value_counts().head(5))
    
    # 7. Create samples for future phases
    # Let's save a clean full dataset, and a smaller 500-review sample for Phase 2 (LLM labeling)
    print("\nSaving processed full dataset to 'cleaned_amazon_reviews.csv'...")
    df_clean.to_csv("cleaned_amazon_reviews.csv", index=False)
    
    # Extract 500 random reviews for our LLM labeling subset in Phase 2
    # Setting random_state=42 ensures reproducibility
    print("Sampling 500 reviews for Phase 2 (LLM labeling) to 'reviews_sample_500.csv'...")
    df_sample = df_clean.sample(n=500, random_state=42)
    df_sample.to_csv("reviews_sample_500.csv", index=False)
    
    print("\nPhase 1 Completed Successfully!")

if __name__ == "__main__":
    run_eda()
