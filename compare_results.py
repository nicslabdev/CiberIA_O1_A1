import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    path = os.path.join(os.getcwd(), 'Data')
    cic_results = np.load(os.path.join(path, 'results_cic.npy'), allow_pickle=True).item()
    unsw_results = np.load(os.path.join(path, 'results_unsw.npy'), allow_pickle=True).item()
    return cic_results, unsw_results

def create_comparison_table(cic_results, unsw_results):
    # Create empty DataFrames for each dataset
    cic_df = pd.DataFrame()
    unsw_df = pd.DataFrame()
    
    # Fill DataFrames
    for preprocessing in cic_results.keys():
        for model, accuracy in cic_results[preprocessing].items():
            cic_df.loc[preprocessing, model] = accuracy
            
    for preprocessing in unsw_results.keys():
        for model, accuracy in unsw_results[preprocessing].items():
            unsw_df.loc[preprocessing, model] = accuracy
    
    # Create MultiIndex for the final table
    datasets = ['CIC-IDS2017', 'UNSW-NB15']
    combined_df = pd.concat([cic_df, unsw_df], keys=datasets)
    
    return combined_df

def plot_heatmap(df):
    plt.figure(figsize=(15, 8))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd')
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'comparison_heatmap.png'))
    plt.close()

def plot_bar_comparison(df):
    # Prepare data for grouped bar plot
    df_reset = df.reset_index()
    df_melted = df_reset.melt(id_vars=['level_0', 'level_1'], 
                             var_name='Model', 
                             value_name='Accuracy')
    
    # Create grouped bar plot
    plt.figure(figsize=(15, 8))
    bar_width = 0.35
    
    # Plot bars for each dataset
    datasets = df_melted['level_0'].unique()
    x = np.arange(len(df.columns))
    
    for i, dataset in enumerate(datasets):
        data = df_melted[df_melted['level_0'] == dataset]
        plt.bar(x + i*bar_width, data.groupby('Model')['Accuracy'].mean(), 
                bar_width, label=dataset)
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison Between Datasets')
    plt.xticks(x + bar_width/2, df.columns, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'comparison_bars.png'))
    plt.close()

def main():
    # Load results
    print("Loading results...")
    cic_results, unsw_results = load_results()
    
    # Create comparison table
    print("\nCreating comparison table...")
    comparison_df = create_comparison_table(cic_results, unsw_results)
    
    # Save table to CSV
    comparison_df.to_csv('model_comparison_results.csv')
    print("\nResults saved to model_comparison_results.csv")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_heatmap(comparison_df)
    plot_bar_comparison(comparison_df)
    
    # Print results
    print("\nFull Comparison Table:")
    print(comparison_df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nMean Accuracy by Dataset:")
    print(comparison_df.groupby(level=0).mean())
    
    print("\nBest Model per Dataset and Preprocessing:")
    for dataset in comparison_df.index.levels[0]:
        print(f"\n{dataset}:")
        for preprocessing in comparison_df.loc[dataset].index:
            best_model = comparison_df.loc[(dataset, preprocessing)].idxmax()
            best_accuracy = comparison_df.loc[(dataset, preprocessing)].max()
            print(f"{preprocessing}: {best_model} ({best_accuracy:.4f})")

if __name__ == "__main__":
    main() 