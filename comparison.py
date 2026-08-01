import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def calculate_metrics(y_true, y_pred, model_name):
    # Calculate precision, recall, f1, and accuracy
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Macro_Precision': precision_macro,
        'Macro_Recall': recall_macro,
        'Macro_F1': f1_macro,
        'Weighted_Precision': precision_weighted,
        'Weighted_Recall': recall_weighted,
        'Weighted_F1': f1_weighted
    }

def main():
    csv_file = "preprocessed_reviews.csv"
    matrix_file = "tfidf_matrix.pkl"
    indices_file = "test_indices.pkl"
    
    # Verify files
    required_files = [csv_file, matrix_file, indices_file]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Missing required file '{f}'. Did you run Phase 3 and 4?")
            sys.exit(1)
            
    print("Loading test split indices and dataset...")
    df = pd.read_csv(csv_file)
    with open(indices_file, 'rb') as f:
        test_indices = pickle.load(f)
    with open(matrix_file, 'rb') as f:
        tfidf_matrix = pickle.load(f)
        
    # Get test subset
    df_test = df.loc[test_indices].copy()
    X_test = tfidf_matrix[test_indices]
    
    # ----------------------------------------------------
    # TASK 1: SENTIMENT COMPARISON (vs. Human Rating)
    # ----------------------------------------------------
    print("\n==========================================")
    print(" TASK 1: SENTIMENT CLASSIFICATION COMPARISON")
    print("==========================================")
    
    # Ground truth is the human star rating sentiment
    y_true_sentiment = df_test['sentiment']
    
    # Load classical models
    sentiment_models = ['naive_bayes', 'logistic_regression', 'random_forest']
    sentiment_results = []
    
    for model_name in sentiment_models:
        model_file = f"sentiment_{model_name}.pkl"
        if not os.path.exists(model_file):
            print(f"Error: Missing model file '{model_file}'. Please run training.py first.")
            sys.exit(1)
            
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        
        display_name = model_name.replace('_', ' ').title()
        sentiment_results.append(calculate_metrics(y_true_sentiment, y_pred, display_name))
        
    # Add LLM zero-shot prediction
    y_pred_llm = df_test['llm_sentiment']
    sentiment_results.append(calculate_metrics(y_true_sentiment, y_pred_llm, "LLM (Zero-Shot)"))
    
    df_sentiment_metrics = pd.DataFrame(sentiment_results)
    print("\nSentiment Results Table (Evaluated against User Star Ratings):")
    print(df_sentiment_metrics[['Model', 'Accuracy', 'Macro_F1', 'Weighted_F1']].to_string(index=False))
    
    # ----------------------------------------------------
    # TASK 2: THEME COMPARISON (vs. LLM Categories)
    # ----------------------------------------------------
    print("\n==========================================")
    print(" TASK 2: THEME/COMPLAINT CLASSIFICATION COMPARISON")
    print("==========================================")
    
    # Target is the LLM-derived categories
    y_true_category = df_test['llm_category']
    category_results = []
    
    category_models = ['naive_bayes', 'logistic_regression', 'random_forest']
    for model_name in category_models:
        model_file = f"category_{model_name}.pkl"
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        
        display_name = model_name.replace('_', ' ').title()
        category_results.append(calculate_metrics(y_true_category, y_pred, display_name))
        
    df_category_metrics = pd.DataFrame(category_results)
    print("\nTheme/Complaint Results Table (Evaluated against LLM Labels):")
    print(df_category_metrics[['Model', 'Accuracy', 'Macro_F1', 'Weighted_F1']].to_string(index=False))
    
    # Export metrics for visualization phase (Phase 6)
    sentiment_metrics_file = "sentiment_metrics_comparison.csv"
    category_metrics_file = "category_metrics_comparison.csv"
    
    df_sentiment_metrics.to_csv(sentiment_metrics_file, index=False)
    df_category_metrics.to_csv(category_metrics_file, index=False)
    
    print(f"\nSaved sentiment comparison to: {os.path.abspath(sentiment_metrics_file)}")
    print(f"Saved theme comparison to: {os.path.abspath(category_metrics_file)}")
    print("\nPhase 5 Completed Successfully!")

if __name__ == "__main__":
    main()
