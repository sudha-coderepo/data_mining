import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_review_text(text):
    # Convert to string in case of non-string values
    text = str(text)
    # Lowercase
    text = text.lower()
    # Remove HTML tags (if any)
    text = re.sub(r'<[^>]*>', '', text)
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    labeled_file = "reviews_labeled_llm.csv"
    if not os.path.exists(labeled_file):
        print(f"Error: Could not find '{labeled_file}'. Did you run Phase 2?")
        # Fallback to cleaned_amazon_reviews.csv if LLM file is missing
        if os.path.exists("cleaned_amazon_reviews.csv"):
            print("Found cleaned_amazon_reviews.csv. Using that as backup for testing...")
            labeled_file = "cleaned_amazon_reviews.csv"
        else:
            return

    print(f"Loading dataset: {labeled_file}...")
    df = pd.read_csv(labeled_file)
    
    # 1. Clean the text column
    print("Cleaning review text (lowercasing, punctuation stripping)...")
    df['cleaned_text'] = df['reviews.text'].apply(clean_review_text)
    
    # Show example of cleaning
    print("\nSample Text Cleaning Comparison:")
    print("Original: ", df['reviews.text'].iloc[0][:100], "...")
    print("Cleaned:  ", df['cleaned_text'].iloc[0][:100], "...")
    
    # 2. Vectorization using TF-IDF (Character N-Grams for multilingual and typo resilience)
    print("\nFitting Character-Level TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape} (reviews, unique_features)")
    
    # 3. Analyze the Vocabulary: Find top terms by average TF-IDF score
    feature_names = np.array(vectorizer.get_feature_names_out())
    mean_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
    sorted_idx = np.argsort(mean_tfidf)[::-1]
    
    print("\nTop 15 Most Important Terms in the Dataset (by average TF-IDF score):")
    for idx, term_idx in enumerate(sorted_idx[:15]):
        print(f"  {idx+1}. {feature_names[term_idx]:<15} (Avg TF-IDF: {mean_tfidf[term_idx]:.4f})")
        
    # 4. Save artifacts for Phase 4
    # Save the cleaned dataframe
    cleaned_export_file = "preprocessed_reviews.csv"
    df.to_csv(cleaned_export_file, index=False)
    print(f"\nSaved preprocessed dataframe to: {os.path.abspath(cleaned_export_file)}")
    
    # Save the fitted TfidfVectorizer and the feature matrix using pickle
    vectorizer_file = "tfidf_vectorizer.pkl"
    matrix_file = "tfidf_matrix.pkl"
    
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(matrix_file, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
        
    print(f"Saved TfidfVectorizer model to: {os.path.abspath(vectorizer_file)}")
    print(f"Saved TF-IDF Feature Matrix to: {os.path.abspath(matrix_file)}")
    print("\nPhase 3 Completed Successfully!")

if __name__ == "__main__":
    main()
