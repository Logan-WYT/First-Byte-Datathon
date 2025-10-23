import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_stadium_sources():
    """Analyze Stadium Operations by Source with monthly and yearly comparisons"""
    print("=== Stadium Operations Source Analysis ===")
    
    # Load stadium data
    stadium_df = pd.read_csv('Byte_Datasets/clean_stadium.csv')
    
    print(f"Loaded stadium data: {stadium_df.shape}")
    print(f"Date range: Month {stadium_df['Month'].min()} to {stadium_df['Month'].max()}")
    
    # Basic source overview
    print(f"\n=== Source Overview ===")
    source_summary = stadium_df.groupby('Source').agg({
        'Revenue': ['sum', 'mean', 'count', 'min', 'max'],
        'Month': ['min', 'max']
    }).round(2)
    
    source_summary.columns = ['Total_Revenue', 'Avg_Revenue', 'Month_Count', 'Min_Revenue', 'Max_Revenue', 'First_Month', 'Last_Month']
    source_summary = source_summary.sort_values('Total_Revenue', ascending=False)
    
    print("Source Summary:")
    print(source_summary)
    source_summary.to_csv('Byte_Datasets/stadium_source_summary.csv')
    
    # Monthly revenue by source
    print(f"\n=== Monthly Revenue Analysis ===")
    monthly_by_source = stadium_df.pivot_table(
        values='Revenue', 
        index='Month', 
        columns='Source', 
        fill_value=0
    )
    
    print("Monthly Revenue by Source (first 6 months):")
    print(monthly_by_source.head(6))
    monthly_by_source.to_csv('Byte_Datasets/monthly_revenue_by_source.csv')
    
    # Calculate yearly totals for each source
    yearly_by_source = monthly_by_source.sum().sort_values(ascending=False)
    print(f"\n=== Yearly Revenue by Source ===")
    print(yearly_by_source)
    yearly_by_source.to_csv('Byte_Datasets/yearly_revenue_by_source.csv')
    
    # Identify sources with zero months
    print(f"\n=== Zero Revenue Months Analysis ===")
    zero_months = (monthly_by_source == 0).sum()
    total_months = len(monthly_by_source)
    zero_percentage = (zero_months / total_months * 100).round(2)
    
    zero_analysis = pd.DataFrame({
        'Source': zero_months.index,
        'Zero_Months': zero_months.values,
        'Total_Months': total_months,
        'Zero_Percentage': zero_percentage.values,
        'Yearly_Revenue': yearly_by_source.values
    }).sort_values('Yearly_Revenue', ascending=False)
    
    print("Sources with zero revenue months:")
    print(zero_analysis)
    zero_analysis.to_csv('Byte_Datasets/zero_months_analysis.csv')
    
    # Create visualizations
    create_stadium_visualizations(monthly_by_source, yearly_by_source, zero_analysis)
    
    # Seasonal analysis
    seasonal_analysis = analyze_seasonality(monthly_by_source)
    
    return monthly_by_source, yearly_by_source, zero_analysis, seasonal_analysis

def analyze_seasonality(monthly_by_source):
    """Analyze seasonal patterns in stadium revenue"""
    print(f"\n=== Seasonal Analysis ===")
    
    # Define seasons
    seasons = {
        'Winter': [1, 2, 12],
        'Spring': [3, 4, 5], 
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11]
    }
    
    seasonal_revenue = {}
    for season_name, months in seasons.items():
        seasonal_data = monthly_by_source.loc[months].sum()
        seasonal_revenue[season_name] = seasonal_data
    
    seasonal_df = pd.DataFrame(seasonal_revenue)
    seasonal_df = seasonal_df.T
    
    print("Seasonal Revenue by Source:")
    print(seasonal_df)
    seasonal_df.to_csv('Byte_Datasets/seasonal_revenue_analysis.csv')
    
    # Find peak and off seasons for each source
    peak_seasons = seasonal_df.idxmax(axis=1)
    off_seasons = seasonal_df.idxmin(axis=1)
    
    seasonality_summary = pd.DataFrame({
        'Peak_Season': peak_seasons,
        'Off_Season': off_seasons,
        'Peak_Revenue': seasonal_df.max(axis=1),
        'Off_Revenue': seasonal_df.min(axis=1),
        'Seasonal_Variation': (seasonal_df.max(axis=1) - seasonal_df.min(axis=1))
    })
    
    print(f"\nSeasonality Summary:")
    print(seasonality_summary)
    seasonality_summary.to_csv('Byte_Datasets/seasonality_summary.csv')
    
    return seasonal_df, seasonality_summary

def create_stadium_visualizations(monthly_by_source, yearly_by_source, zero_analysis):
    """Create comprehensive visualizations for stadium analysis"""
    
    # Set up the plotting style
    plt.style.use('default')
    
    # 1. Yearly Revenue by Source (Bar Chart)
    plt.figure(figsize=(14, 8))
    yearly_by_source.plot(kind='bar', color='steelblue', alpha=0.8)
    plt.title('Yearly Revenue by Source', fontsize=16, fontweight='bold')
    plt.xlabel('Source', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/yearly_revenue_by_source.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monthly Revenue Heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(monthly_by_source.T, annot=True, fmt='.0f', cmap='RdYlBu_r', 
                center=0, cbar_kws={'label': 'Revenue ($)'})
    plt.title('Monthly Revenue Heatmap by Source', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Source', fontsize=12)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/monthly_revenue_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Revenue Sources with Zero Months
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(zero_analysis)), zero_analysis['Zero_Percentage'], 
                   color=['red' if x > 50 else 'orange' if x > 25 else 'green' for x in zero_analysis['Zero_Percentage']])
    plt.title('Percentage of Zero Revenue Months by Source', fontsize=16, fontweight='bold')
    plt.xlabel('Source', fontsize=12)
    plt.ylabel('Percentage of Zero Months (%)', fontsize=12)
    plt.xticks(range(len(zero_analysis)), zero_analysis['Source'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/zero_months_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Top Revenue Sources - Monthly Trends
    top_sources = yearly_by_source.head(6).index
    plt.figure(figsize=(14, 8))
    
    for source in top_sources:
        plt.plot(monthly_by_source.index, monthly_by_source[source], 
                marker='o', linewidth=2, label=source, markersize=6)
    
    plt.title('Monthly Revenue Trends - Top 6 Sources', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/top_sources_monthly_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Revenue Distribution by Source
    plt.figure(figsize=(12, 8))
    monthly_by_source.boxplot(figsize=(12, 8))
    plt.title('Revenue Distribution by Source (Monthly)', fontsize=16, fontweight='bold')
    plt.xlabel('Source', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/revenue_distribution_by_source.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved all stadium visualizations")

def create_comparison_table(monthly_by_source, yearly_by_source):
    """Create a comprehensive comparison table"""
    print(f"\n=== Comprehensive Source Comparison ===")
    
    comparison_data = []
    
    for source in yearly_by_source.index:
        monthly_data = monthly_by_source[source]
        
        # Calculate metrics
        yearly_total = yearly_by_source[source]
        monthly_avg = monthly_data.mean()
        monthly_std = monthly_data.std()
        zero_months = (monthly_data == 0).sum()
        positive_months = (monthly_data > 0).sum()
        negative_months = (monthly_data < 0).sum()
        max_month = monthly_data.max()
        min_month = monthly_data.min()
        consistency_score = positive_months / 12 * 100
        
        comparison_data.append({
            'Source': source,
            'Yearly_Revenue': yearly_total,
            'Monthly_Average': monthly_avg,
            'Monthly_StdDev': monthly_std,
            'Zero_Months': zero_months,
            'Positive_Months': positive_months,
            'Negative_Months': negative_months,
            'Max_Monthly': max_month,
            'Min_Monthly': min_month,
            'Consistency_Score_%': consistency_score
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Yearly_Revenue', ascending=False)
    
    print("Comprehensive Source Comparison:")
    print(comparison_df.round(2))
    comparison_df.to_csv('Byte_Datasets/comprehensive_source_comparison.csv', index=False)
    
    return comparison_df

def main():
    """Main analysis function"""
    print("Starting Stadium Operations Source Analysis...")
    
    # Run main analysis
    monthly_by_source, yearly_by_source, zero_analysis, seasonal_data = analyze_stadium_sources()
    
    # Create comprehensive comparison
    comparison_df = create_comparison_table(monthly_by_source, yearly_by_source)
    
    # Key insights
    print(f"\n=== KEY INSIGHTS ===")
    
    # Top performers
    top_3_yearly = yearly_by_source.head(3)
    print(f"Top 3 Revenue Sources (Yearly):")
    for source, revenue in top_3_yearly.items():
        print(f"  {source}: ${revenue:,.0f}")
    
    # Most consistent sources
    most_consistent = comparison_df.nlargest(3, 'Consistency_Score_%')
    print(f"\nMost Consistent Sources (Fewest Zero Months):")
    for _, row in most_consistent.iterrows():
        print(f"  {row['Source']}: {row['Consistency_Score_%']:.1f}% consistency")
    
    # Seasonal sources
    seasonal_sources = zero_analysis[zero_analysis['Zero_Percentage'] > 50]
    print(f"\nHighly Seasonal Sources (>50% zero months):")
    for _, row in seasonal_sources.iterrows():
        print(f"  {row['Source']}: {row['Zero_Percentage']:.1f}% zero months")
    
    print(f"\n✓ Analysis complete! Check Byte_Datasets/ for all outputs and figures/")

if __name__ == "__main__":
    main()
