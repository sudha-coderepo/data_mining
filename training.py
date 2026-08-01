import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_and_evaluate_task(X, y, task_name, model_save_prefix):
    print(f"\n==========================================")
    print(f" TRAINING MODELS FOR: {task_name.upper()}")
    print(f"==========================================")
    
    # Split data (80% training / 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size:  {X_test.shape[0]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # 1. Naive Bayes
    print("\nTraining Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    nb_report = classification_report(y_test, y_pred_nb, zero_division=0)
    print("--- Naive Bayes Classification Report ---")
    print(nb_report)
    
    # 2. Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_report = classification_report(y_test, y_pred_lr, zero_division=0)
    print("--- Logistic Regression Classification Report ---")
    print(lr_report)
    
    # 3. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_report = classification_report(y_test, y_pred_rf, zero_division=0)
    print("--- Random Forest Classification Report ---")
    print(rf_report)
    
    # Save the models
    models = {
        'naive_bayes': nb_model,
        'logistic_regression': lr_model,
        'random_forest': rf_model
    }
    
    for model_name, model_obj in models.items():
        filename = f"{model_save_prefix}_{model_name}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model_obj, f)
        print(f"Saved {model_name} model to: {os.path.abspath(filename)}")

def main():
    csv_file = "preprocessed_reviews.csv"
    matrix_file = "tfidf_matrix.pkl"
    
    # Check if files exist
    if not os.path.exists(csv_file) or not os.path.exists(matrix_file):
        print("Error: Missing preprocessed files. Please run preprocessing.py first.")
        sys.exit(1)
        
    print("Loading preprocessed dataset and TF-IDF matrix...")
    df = pd.read_csv(csv_file)
    with open(matrix_file, 'rb') as f:
        tfidf_matrix = pickle.load(f)
        
    print(f"Dataset shape: {df.shape}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Verify the LLM columns are present
    if 'llm_sentiment' not in df.columns or 'llm_category' not in df.columns:
        print("Error: Labeled columns ('llm_sentiment', 'llm_category') not found in CSV.")
        sys.exit(1)
        
    # Task 1: Sentiment Classification
    # Train classifiers to learn the LLM-derived sentiment labels
    train_and_evaluate_task(
        X=tfidf_matrix,
        y=df['llm_sentiment'],
        task_name="Sentiment Classification",
        model_save_prefix="sentiment"
    )
    
    # Task 2: Theme / Category Classification
    # Train classifiers to learn the LLM-derived complaint/request categories
    train_and_evaluate_task(
        X=tfidf_matrix,
        y=df['llm_category'],
        task_name="Theme/Complaint Classification",
        model_save_prefix="category"
    )
    
    # Save the test indices so that Phase 5 comparison uses the exact same test subset
    test_indices_file = "test_indices.pkl"
    # We re-run the split logic to extract indices
    _, test_indices = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=df['llm_category']
    )
    with open(test_indices_file, 'wb') as f:
        pickle.dump(test_indices, f)
    print(f"\nSaved test split indices to: {os.path.abspath(test_indices_file)}")
    
    print("\nPhase 4 Completed Successfully!")

if __name__ == "__main__":
    main()
