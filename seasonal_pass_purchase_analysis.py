import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def merge_and_analyze_seasonal_pass_purchases():
    """Merge Merchandise and Fanbase datasets and analyze seasonal pass impact on purchases"""
    print("=== Seasonal Pass vs Purchase Behavior Analysis ===")
    
    # Load datasets
    merch_df = pd.read_csv('Byte_Datasets/clean_merch.csv')
    fanbase_df = pd.read_csv('Byte_Datasets/clean_fanbase.csv')
    
    print(f"Merchandise data: {merch_df.shape}")
    print(f"Fanbase data: {fanbase_df.shape}")
    
    # Check Member_ID overlap
    merch_members = set(merch_df['Member_ID'].unique())
    fanbase_members = set(fanbase_df['Membership_ID'].unique())
    common_members = merch_members.intersection(fanbase_members)
    
    print(f"\nMember ID Analysis:")
    print(f"  Merchandise unique members: {len(merch_members):,}")
    print(f"  Fanbase unique members: {len(fanbase_members):,}")
    print(f"  Common members: {len(common_members):,}")
    print(f"  Overlap percentage: {len(common_members)/len(merch_members)*100:.1f}%")
    
    # Merge datasets on Member_ID
    print(f"\n=== Merging Datasets ===")
    merged_df = merch_df.merge(
        fanbase_df, 
        left_on='Member_ID', 
        right_on='Membership_ID', 
        how='inner'
    )
    
    print(f"Merged dataset: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")
    
    # Analyze seasonal pass impact on purchases
    analyze_seasonal_pass_impact(merged_df)
    
    return merged_df

def analyze_seasonal_pass_impact(df):
    """Analyze the impact of seasonal pass on purchasing behavior"""
    print(f"\n=== Seasonal Pass Impact Analysis ===")
    
    # Basic statistics by seasonal pass status
    seasonal_pass_stats = df.groupby('Seasonal_Pass').agg({
        'Unit_Price': ['count', 'sum', 'mean', 'std'],
        'Member_ID': 'nunique',
        'Games_Attended': 'mean'
    }).round(2)
    
    print("Purchase Statistics by Seasonal Pass Status:")
    print(seasonal_pass_stats)
    seasonal_pass_stats.to_csv('Byte_Datasets/seasonal_pass_purchase_stats.csv')
    
    # Calculate per-member metrics
    member_purchase_stats = df.groupby(['Member_ID', 'Seasonal_Pass']).agg({
        'Unit_Price': ['count', 'sum', 'mean'],
        'Games_Attended': 'first',  # Should be same for all rows per member
        'Age_Group': 'first',
        'Customer_Region_y': 'first'  # Use the fanbase region column
    }).round(2)
    
    # Flatten column names
    member_purchase_stats.columns = ['Purchase_Count', 'Total_Spent', 'Avg_Purchase', 'Games_Attended', 'Age_Group', 'Region']
    member_purchase_stats = member_purchase_stats.reset_index()
    
    print(f"\nPer-Member Purchase Statistics:")
    print(member_purchase_stats.groupby('Seasonal_Pass')[['Purchase_Count', 'Total_Spent', 'Avg_Purchase', 'Games_Attended']].mean())
    
    # Detailed analysis
    detailed_analysis = member_purchase_stats.groupby('Seasonal_Pass').agg({
        'Purchase_Count': ['count', 'mean', 'std', 'min', 'max'],
        'Total_Spent': ['mean', 'std', 'min', 'max'],
        'Avg_Purchase': ['mean', 'std'],
        'Games_Attended': ['mean', 'std']
    }).round(2)
    
    print(f"\nDetailed Per-Member Analysis:")
    print(detailed_analysis)
    detailed_analysis.to_csv('Byte_Datasets/seasonal_pass_detailed_analysis.csv')
    
    # Category analysis by seasonal pass
    category_analysis = df.groupby(['Seasonal_Pass', 'Item_Category']).agg({
        'Unit_Price': ['count', 'sum', 'mean']
    }).round(2)
    
    print(f"\nCategory Analysis by Seasonal Pass:")
    print(category_analysis)
    category_analysis.to_csv('Byte_Datasets/seasonal_pass_category_analysis.csv')
    
    # Age group analysis
    age_analysis = member_purchase_stats.groupby(['Seasonal_Pass', 'Age_Group']).agg({
        'Purchase_Count': 'mean',
        'Total_Spent': 'mean',
        'Games_Attended': 'mean'
    }).round(2)
    
    print(f"\nAge Group Analysis by Seasonal Pass:")
    print(age_analysis)
    age_analysis.to_csv('Byte_Datasets/seasonal_pass_age_analysis.csv')
    
    # Create visualizations
    create_seasonal_pass_visualizations(member_purchase_stats, df)

def create_seasonal_pass_visualizations(member_stats, transaction_df):
    """Create visualizations for seasonal pass analysis"""
    
    # 1. Purchase Count Distribution
    plt.figure(figsize=(14, 8))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Purchase Count Comparison
    purchase_counts = [member_stats[member_stats['Seasonal_Pass'] == 'FALSE']['Purchase_Count'],
                      member_stats[member_stats['Seasonal_Pass'] == 'TRUE']['Purchase_Count']]
    
    axes[0, 0].boxplot(purchase_counts, labels=['No Seasonal Pass', 'Seasonal Pass'])
    axes[0, 0].set_title('Purchase Count Distribution by Seasonal Pass Status')
    axes[0, 0].set_ylabel('Number of Purchases')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Total Spending Comparison
    total_spent = [member_stats[member_stats['Seasonal_Pass'] == 'FALSE']['Total_Spent'],
                   member_stats[member_stats['Seasonal_Pass'] == 'TRUE']['Total_Spent']]
    
    axes[0, 1].boxplot(total_spent, labels=['No Seasonal Pass', 'Seasonal Pass'])
    axes[0, 1].set_title('Total Spending Distribution by Seasonal Pass Status')
    axes[0, 1].set_ylabel('Total Spent ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average Purchase Comparison
    avg_purchase = [member_stats[member_stats['Seasonal_Pass'] == 'FALSE']['Avg_Purchase'],
                    member_stats[member_stats['Seasonal_Pass'] == 'TRUE']['Avg_Purchase']]
    
    axes[1, 0].boxplot(avg_purchase, labels=['No Seasonal Pass', 'Seasonal Pass'])
    axes[1, 0].set_title('Average Purchase Amount by Seasonal Pass Status')
    axes[1, 0].set_ylabel('Average Purchase ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Games Attended vs Purchases Scatter
    for pass_status in ['FALSE', 'TRUE']:
        data = member_stats[member_stats['Seasonal_Pass'] == pass_status]
        axes[1, 1].scatter(data['Games_Attended'], data['Purchase_Count'], 
                          label=f'Seasonal Pass: {pass_status}', alpha=0.6)
    
    axes[1, 1].set_title('Games Attended vs Purchase Count')
    axes[1, 1].set_xlabel('Games Attended')
    axes[1, 1].set_ylabel('Purchase Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/seasonal_pass_purchase_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Category Preferences by Seasonal Pass
    plt.figure(figsize=(14, 8))
    category_purchase_counts = transaction_df.groupby(['Seasonal_Pass', 'Item_Category']).size().unstack(fill_value=0)
    
    category_purchase_counts.plot(kind='bar', figsize=(14, 8), width=0.8)
    plt.title('Purchase Count by Category and Seasonal Pass Status')
    plt.xlabel('Seasonal Pass Status')
    plt.ylabel('Number of Purchases')
    plt.legend(title='Item Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/seasonal_pass_category_preferences.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Age Group Analysis
    plt.figure(figsize=(12, 8))
    age_purchase_analysis = member_stats.groupby(['Seasonal_Pass', 'Age_Group']).agg({
        'Purchase_Count': 'mean',
        'Total_Spent': 'mean'
    }).unstack()
    
    age_purchase_analysis['Purchase_Count'].plot(kind='bar', figsize=(12, 8))
    plt.title('Average Purchase Count by Age Group and Seasonal Pass Status')
    plt.xlabel('Age Group')
    plt.ylabel('Average Purchase Count')
    plt.legend(title='Seasonal Pass Status')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/seasonal_pass_age_group_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary Statistics Bar Chart
    plt.figure(figsize=(12, 8))
    
    summary_stats = member_stats.groupby('Seasonal_Pass').agg({
        'Purchase_Count': 'mean',
        'Total_Spent': 'mean',
        'Avg_Purchase': 'mean',
        'Games_Attended': 'mean'
    })
    
    # Normalize for comparison (scale to 0-1)
    normalized_stats = summary_stats.div(summary_stats.max()).T
    
    normalized_stats.plot(kind='bar', figsize=(12, 8))
    plt.title('Normalized Purchase Behavior by Seasonal Pass Status')
    plt.xlabel('Metrics')
    plt.ylabel('Normalized Value (0-1)')
    plt.legend(title='Seasonal Pass Status')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/seasonal_pass_summary_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ All seasonal pass visualizations saved")

def calculate_purchase_lift():
    """Calculate the lift in purchasing behavior for seasonal pass holders"""
    print(f"\n=== Purchase Lift Analysis ===")
    
    # Calculate directly from the data instead of loading CSV
    # Load the merged data
    merch_df = pd.read_csv('Byte_Datasets/clean_merch.csv')
    fanbase_df = pd.read_csv('Byte_Datasets/clean_fanbase.csv')
    
    merged_df = merch_df.merge(fanbase_df, left_on='Member_ID', right_on='Membership_ID', how='inner')
    
    # Calculate per-member metrics
    member_stats = merged_df.groupby(['Member_ID', 'Seasonal_Pass']).agg({
        'Unit_Price': ['count', 'sum', 'mean']
    }).round(2)
    
    member_stats.columns = ['Purchase_Count', 'Total_Spent', 'Avg_Purchase']
    member_stats = member_stats.reset_index()
    
    # Calculate averages by seasonal pass status
    no_pass_stats = member_stats[member_stats['Seasonal_Pass'] == 'FALSE'].mean()
    pass_stats = member_stats[member_stats['Seasonal_Pass'] == 'TRUE'].mean()
    
    no_pass_purchase_count = no_pass_stats['Purchase_Count']
    pass_purchase_count = pass_stats['Purchase_Count']
    
    no_pass_total_spent = no_pass_stats['Total_Spent']
    pass_total_spent = pass_stats['Total_Spent']
    
    no_pass_avg_purchase = no_pass_stats['Avg_Purchase']
    pass_avg_purchase = pass_stats['Avg_Purchase']
    
    # Calculate lift percentages
    purchase_count_lift = ((pass_purchase_count - no_pass_purchase_count) / no_pass_purchase_count) * 100
    total_spent_lift = ((pass_total_spent - no_pass_total_spent) / no_pass_total_spent) * 100
    avg_purchase_lift = ((pass_avg_purchase - no_pass_avg_purchase) / no_pass_avg_purchase) * 100
    
    print(f"Purchase Behavior Lift for Seasonal Pass Holders:")
    print(f"  Purchase Count: {purchase_count_lift:+.1f}%")
    print(f"  Total Spent: {total_spent_lift:+.1f}%")
    print(f"  Average Purchase: {avg_purchase_lift:+.1f}%")
    
    # Save lift analysis
    lift_data = {
        'Metric': ['Purchase Count', 'Total Spent', 'Average Purchase'],
        'No Pass': [no_pass_purchase_count, no_pass_total_spent, no_pass_avg_purchase],
        'With Pass': [pass_purchase_count, pass_total_spent, pass_avg_purchase],
        'Lift %': [purchase_count_lift, total_spent_lift, avg_purchase_lift]
    }
    
    lift_df = pd.DataFrame(lift_data)
    print(f"\nLift Analysis Summary:")
    print(lift_df.round(2))
    lift_df.to_csv('Byte_Datasets/seasonal_pass_lift_analysis.csv', index=False)

def main():
    """Main function to run seasonal pass analysis"""
    merged_df = merge_and_analyze_seasonal_pass_purchases()
    calculate_purchase_lift()
    
    print(f"\n=== SEASONAL PASS PURCHASE ANALYSIS COMPLETE ===")
    print("All analysis files saved to Byte_Datasets/")
    print("All visualizations saved to Byte_Datasets/figures/merchandise_sales/")
    print(f"Merged dataset shape: {merged_df.shape}")

if __name__ == "__main__":
    main()
