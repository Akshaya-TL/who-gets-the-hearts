import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_ai_vs_human_comments(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)

    print("===========================================")
    print(" 1. CATEGORY PERCENTAGE BREAKDOWN")
    print("===========================================")
    # Calculate the percentage of each category within the AI and Human conditions
    category_pct = pd.crosstab(df['category'], df['condition'], normalize='columns') * 100
    print(category_pct.round(2).to_string())

    print("\n===========================================")
    print(" 2. CHI-SQUARE TEST OF INDEPENDENCE")
    print("===========================================")
    # Create a raw count cross-tabulation for the statistical test
    category_counts = pd.crosstab(df['category'], df['condition'])
    
    chi2, p, dof, expected = stats.chi2_contingency(category_counts)
    print(f"Chi-Square Statistic: {chi2:.2f}")
    print(f"P-value: {p:.2e}")
    
    if p < 0.05:
        print("Result: SIGNIFICANT DIFFERENCE. AI and Human posts receive fundamentally different types of comments.")
    else:
        print("Result: No significant difference in comment distributions.")

    print("\n===========================================")
    print(" 3. CONFIDENCE LEVEL ANALYSIS")
    print("===========================================")
    # Map confidence to a numerical scale
    conf_map = {'low': 1, 'medium': 2, 'high': 3}
    df['conf_score'] = df['confidence'].map(conf_map)
    
    # Drop rows with missing confidence scores just in case
    df_clean = df.dropna(subset=['conf_score'])
    
    ai_conf = df_clean[df_clean['condition'] == 'AI']['conf_score']
    human_conf = df_clean[df_clean['condition'] == 'Human']['conf_score']
    
    print(f"Mean Confidence (AI): {ai_conf.mean():.2f}")
    print(f"Mean Confidence (Human): {human_conf.mean():.2f}")
    
    # Independent T-test
    t_stat, p_val = stats.ttest_ind(ai_conf, human_conf, equal_var=False)
    print(f"T-test P-value for Confidence Difference: {p_val:.4f}")

    print("\n===========================================")
    print(" 4. GENERATING VISUALIZATION...")
    print("===========================================")
    # Create a dataframe specifically for percentage plotting
    pct_df = df.groupby(['condition', 'category']).size().reset_index(name='count')
    total_per_condition = pct_df.groupby('condition')['count'].transform('sum')
    pct_df['percentage'] = (pct_df['count'] / total_per_condition) * 100

    # Plot the results side-by-side
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=pct_df, 
        x='category', 
        y='percentage',
        hue='condition', 
        order=df['category'].value_counts().index
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Percentage of Comment Categories: AI vs. Human')
    plt.ylabel('Percentage of Comments (%)')
    plt.xlabel('Comment Category')
    plt.tight_layout()
    plt.savefig('category_comparison.png')
    print("Saved plot as 'category_comparison.png'")

if __name__ == "__main__":
    analyze_ai_vs_human_comments("classified_comments.csv")
