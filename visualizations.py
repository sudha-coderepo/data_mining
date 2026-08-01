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

def main():
    print("Generating performance comparison charts...")
    plot_sentiment_metrics()
    plot_category_metrics()
    print("\nPhase 6 Visualizations Generated Successfully!")

if __name__ == "__main__":
    main()
