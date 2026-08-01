import zipfile
import pandas as pd
import os

def main():
    zip_path = "C:/Users/Sudha/Downloads/archive (1).zip"
    if not os.path.exists(zip_path):
        print(f"Error: Could not find zip file at {zip_path}")
        return

    print("Opening ZIP archive...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        print("Loading 1429_1.csv into memory...")
        # Load the CSV
        df_raw = pd.read_csv(z.open('1429_1.csv'), low_memory=False)

    print(f"Successfully loaded {len(df_raw):,} reviews.")

    # 1. Drop rows with null ratings or review text
    df_raw = df_raw.dropna(subset=['reviews.rating', 'reviews.text']).copy()
    print(f"Rows remaining after removing nulls: {len(df_raw):,}")

    # 2. Separate into sentiment categories
    # 1-2 stars: Negative
    # 3 stars: Neutral
    # 4-5 stars: Positive
    df_neg = df_raw[df_raw['reviews.rating'] <= 2.0].copy()
    df_neu = df_raw[df_raw['reviews.rating'] == 3.0].copy()
    df_pos = df_raw[df_raw['reviews.rating'] >= 4.0].copy()

    neg_count = len(df_neg)
    neu_count = len(df_neu)
    pos_count = len(df_pos)
    
    print(f"\nAvailable counts in raw data:")
    print(f"  Negative (1-2 stars): {neg_count}")
    print(f"  Neutral (3 stars):    {neu_count}")
    print(f"  Positive (4-5 stars): {pos_count}")

    # 3. Create a balanced sample
    # Take all negatives, all neutrals, and sample 2,500 positives
    print("\nCreating balanced 3-class sample...")
    df_pos_sampled = df_pos.sample(n=2500, random_state=42)

    # Combine them
    df_balanced = pd.concat([df_neg, df_neu, df_pos_sampled], ignore_index=True)
    
    # Shuffle the dataset so positive/neutral/negative are mixed
    df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 4. Map numerical rating to sentiment label
    def map_sentiment(rating):
        if rating >= 4.0:
            return "Positive"
        elif rating == 3.0:
            return "Neutral"
        else:
            return "Negative"

    df_balanced['sentiment'] = df_balanced['reviews.rating'].apply(map_sentiment)
    
    # 5. Extract only relevant columns to keep it clean and match previous schemas
    columns_to_keep = ['id', 'reviews.rating', 'reviews.text', 'reviews.title', 'sentiment']
    # If id doesn't exist or has nulls, create one
    if 'id' not in df_balanced.columns:
        df_balanced['id'] = [f"AMZN_{i:05d}" for i in range(len(df_balanced))]
    else:
        df_balanced['id'] = df_balanced['id'].fillna("").apply(lambda x: x if x else f"AMZN_{random.randint(10000,99999)}")
        
    df_final = df_balanced[columns_to_keep].copy()

    print("\n--- Summary of Balanced Dataset ---")
    print(f"Total reviews: {len(df_final)}")
    print("\nSentiment Class Distribution:")
    print(df_final['sentiment'].value_counts())

    # Save the balanced reviews
    target_csv = "cleaned_amazon_reviews.csv"
    df_final.to_csv(target_csv, index=False)
    print(f"\nSaved balanced dataset to: {os.path.abspath(target_csv)}")

    # Save a 500-review sample for quick testing
    sample_csv = "reviews_sample_500.csv"
    df_final.sample(n=500, random_state=42).to_csv(sample_csv, index=False)
    print(f"Saved 500-review sample to: {os.path.abspath(sample_csv)}")
    
    print("\nExtraction & Balancing Completed Successfully!")

if __name__ == "__main__":
    main()
