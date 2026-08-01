import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_metrics():
    file_path = "sentiment_metrics_comparison.csv"
    if not os.path.exists(file_path):
        print(f"Error: Missing '{file_path}'. Please run comparison.py first.")
        return
        
    df = pd.read_csv(file_path)
    
    # Format the data for plotting
    df_melted = df.melt(
        id_vars=['Model'], 
        value_vars=['Accuracy', 'Macro_F1', 'Weighted_F1'],
        var_name='Metric', 
        value_name='Score'
    )
    
    # Set professional style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Modern color palette
    colors = ["#2b7bba", "#e68422", "#4caf50"]
    
    ax = sns.barplot(
        data=df_melted, 
        x='Model', 
        y='Score', 
        hue='Metric', 
        palette=colors,
        edgecolor='black',
        linewidth=0.8
    )
    
    # Title and Labels
    plt.title("Sentiment Classification Model Comparison\n(Evaluated against User Ratings Ground Truth)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Classification Model", fontsize=12, labelpad=10)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12, labelpad=10)
    plt.ylim(0, 1.05)
    
    # Annotate bars with values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # skip labels for 0 height bars
            ax.annotate(
                f'{height:.2f}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', 
                va='center', 
                xytext=(0, 8), 
                textcoords='offset points', 
                fontsize=9, 
                fontweight='semibold'
            )
            
    plt.legend(title="Metric Type", loc='lower right', frameon=True)
    plt.tight_layout()
    
    output_img = "sentiment_comparison.png"
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"Saved sentiment comparison plot to: {os.path.abspath(output_img)}")

def plot_category_metrics():
    file_path = "category_metrics_comparison.csv"
    if not os.path.exists(file_path):
        print(f"Error: Missing '{file_path}'. Please run comparison.py first.")
        return
        
    df = pd.read_csv(file_path)
    
    # Format the data for plotting
    df_melted = df.melt(
        id_vars=['Model'], 
        value_vars=['Accuracy', 'Macro_F1', 'Weighted_F1'],
        var_name='Metric', 
        value_name='Score'
    )
    
    # Set professional style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Modern color palette
    colors = ["#4a148c", "#00838f", "#ad1457"]
    
    ax = sns.barplot(
        data=df_melted, 
        x='Model', 
        y='Score', 
        hue='Metric', 
        palette=colors,
        edgecolor='black',
        linewidth=0.8
    )
    
    # Title and Labels
    plt.title("Theme/Complaint Classification Model Comparison\n(Evaluated against LLM-Generated Target Labels)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Classification Model", fontsize=12, labelpad=10)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12, labelpad=10)
    plt.ylim(0, 1.05)
    
    # Annotate bars with values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f'{height:.2f}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', 
                va='center', 
                xytext=(0, 8), 
                textcoords='offset points', 
                fontsize=9, 
                fontweight='semibold'
            )
            
    plt.legend(title="Metric Type", loc='lower right', frameon=True)
    plt.tight_layout()
    
    output_img = "category_comparison.png"
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"Saved category comparison plot to: {os.path.abspath(output_img)}")

def plot_confusion_matrices():
    import pickle
    from sklearn.metrics import confusion_matrix
    
    csv_file = "preprocessed_reviews.csv"
    matrix_file = "tfidf_matrix.pkl"
    indices_file = "test_indices.pkl"
    
    if not (os.path.exists(csv_file) and os.path.exists(matrix_file) and os.path.exists(indices_file)):
        print("Error: Missing required files to plot confusion matrices.")
        return
        
    df = pd.read_csv(csv_file)
    with open(indices_file, "rb") as f:
        test_indices = pickle.load(f)
    with open(matrix_file, "rb") as f:
        tfidf_matrix = pickle.load(f)
        
    df_test = df.loc[test_indices]
    X_test = tfidf_matrix[test_indices]
    
    # 1. Sentiment Confusion Matrix
    if os.path.exists("sentiment_random_forest.pkl"):
        with open("sentiment_random_forest.pkl", "rb") as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        y_true = df_test['llm_sentiment']
        
        plt.figure(figsize=(6, 5))
        classes = sorted(list(set(y_true)))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
        plt.title("Sentiment Confusion Matrix\n(Random Forest on Holdout Test Set)", fontsize=12, fontweight='bold', pad=10)
        plt.ylabel("True Label", fontsize=10)
        plt.xlabel("Predicted Label", fontsize=10)
        plt.tight_layout()
        
        output_img = "sentiment_confusion.png"
        plt.savefig(output_img, dpi=300)
        plt.close()
        print(f"Saved sentiment confusion matrix to: {os.path.abspath(output_img)}")
        
    # 2. Theme Confusion Matrix
    if os.path.exists("category_random_forest.pkl"):
        with open("category_random_forest.pkl", "rb") as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        y_true = df_test['llm_category']
        
        plt.figure(figsize=(8, 6))
        classes = sorted(list(set(y_true)))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes, cbar=False)
        plt.title("Theme Confusion Matrix\n(Random Forest on Holdout Test Set)", fontsize=12, fontweight='bold', pad=10)
        plt.ylabel("True Label", fontsize=10)
        plt.xlabel("Predicted Label", fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_img = "category_confusion.png"
        plt.savefig(output_img, dpi=300)
        plt.close()
        print(f"Saved theme confusion matrix to: {os.path.abspath(output_img)}")

def main():
    print("Generating performance comparison charts...")
    plot_sentiment_metrics()
    plot_category_metrics()
    print("Generating confusion matrix heatmaps...")
    plot_confusion_matrices()
    print("\nPhase 6 Visualizations Generated Successfully!")

if __name__ == "__main__":
    main()
