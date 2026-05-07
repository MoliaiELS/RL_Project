import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 200})

def analyze_csv(csv_path,tag):
    print(f"\n{'='*30}")
    print(f" Analysis for {tag.upper()}")
    print(f"Reading path: {csv_path}")
    
    # Load the results
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Filter out failed runs
    df = df[df['error'].isna()]
    
    
    # Ensure numerical types
    df['eval_mean_reward'] = pd.to_numeric(df['eval_mean_reward'])
    valid_df = df.dropna(subset=['eval_mean_reward'])
    
    if valid_df.empty:
        print("\n[Warning] CSV (after filtering) contains no valid evaluation rewards. Cannot perform analysis.")
        print("Please ensure you have re-run 'python scripts/param_search.py --eval-after-train' and the evaluation was successful.")
        return
    
    
    # Create results directory
    output_dir = os.path.join("eval_results", tag.lower())
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for analysis: {output_dir}")

    # 1. TOP PERFORMERS PER ENVIRONMENT
    print("\n--- Top Performers per Environment ---")
    # Get the row with the maximum eval_mean_reward for each env_id
    best_runs = valid_df.sort_values('eval_mean_reward', ascending=False).drop_duplicates('env_id')
    print(best_runs[['env_id', 'alpha', 'lambda_value', 'eval_mean_reward']])
    
    # Export to LaTeX Table for the report
    try:
        latex_table = best_runs[['env_id', 'method', 'alpha', 'lambda_value', 'eval_mean_reward']].to_latex(index=False)
        with open(os.path.join(output_dir, "best_params_table.tex"), "w", encoding='utf-8') as f:
            f.write(latex_table)
    except:
        print("Note: LaTeX export failed, but plots will continue.")

    # 2. LAMBDA SENSITIVITY ANALYSIS (Line Plot)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='lambda_value', y='eval_mean_reward', hue='env_id', marker='o', errorbar='sd')
    plt.title("Impact of Lambda on Evaluation Reward", fontsize=14)
    plt.xlabel("λ (Lambda Value)", fontsize=12)
    plt.ylabel("Mean Evaluation Reward", fontsize=12)
    plt.legend(title="Environment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(output_dir, "lambda_sensitivity.png"))
    plt.close()

    # 3. ALPHA-LAMBDA INTERACTION (Heatmap)
    # We choose the most complex environment for this visualization
    envs = df['env_id'].unique()
    for env in envs:
        subset = df[df['env_id'] == env]
        if subset['alpha'].nunique() > 1 and subset['lambda_value'].nunique() > 1:
            # Pivot table for heatmap
            pivot = subset.pivot_table(index='alpha', columns='lambda_value', values='eval_mean_reward')
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".3f")
            plt.title(f"Alpha-Lambda Interaction: {env}")
            plt.xlabel("λ Value")
            plt.ylabel("Learning Rate (Alpha)")
            plt.savefig(os.path.join(output_dir, f"heatmap_{env.replace('/', '_')}.png"), bbox_inches='tight')
            plt.close()

    # 4. ALPHA SENSITIVITY (Box Plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='alpha', y='eval_mean_reward', hue='method')
    plt.title("Learning Rate (Alpha) Stability", fontsize=14)
    plt.xlabel("Learning Rate", fontsize=12)
    plt.ylabel("Evaluation Reward", fontsize=12)
    plt.savefig(os.path.join(output_dir, "alpha_stability.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Point this to your actual CSV output path
    BASE_CSV = os.path.join("saved_models", "base_tdλ", "search_summary.csv")
    FEAT_CSV = os.path.join("saved_models", "action_features", "search_summary.csv")
    RANDOM_BASE_CSV = os.path.join("saved_models_random", "base_tdλ", "search_summary.csv")
    RANDOM_FEAT_CSV = os.path.join("saved_models_random", "action_features", "search_summary.csv")
    
    analyze_csv(BASE_CSV, "base")
    analyze_csv(FEAT_CSV, "action_features")
    analyze_csv(RANDOM_BASE_CSV, "random_base")
    analyze_csv(RANDOM_FEAT_CSV, "random_action_features")