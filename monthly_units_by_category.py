import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_monthly_units_by_category():
    """Create graph showing monthly units sold by category"""
    print("=== Monthly Units Sold by Category Analysis ===")
    
    # Load merchandise data
    merch_df = pd.read_csv('Byte_Datasets/clean_merch.csv')
    
    print(f"Loaded merchandise data: {merch_df.shape}")
    
    # Convert Selling_Date to datetime and extract month
    merch_df['Selling_Date'] = pd.to_datetime(merch_df['Selling_Date'])
    merch_df['Month'] = merch_df['Selling_Date'].dt.month
    
    # Count units sold by category and month
    monthly_units_by_category = merch_df.groupby(['Month', 'Item_Category']).size().unstack(fill_value=0)
    
    print("Monthly Units Sold by Category:")
    print(monthly_units_by_category)
    
    # Save data
    monthly_units_by_category.to_csv('Byte_Datasets/monthly_units_by_category.csv')
    
    # Create visualizations
    create_monthly_units_visualizations(monthly_units_by_category)
    
    print("✓ Monthly units analysis completed!")

def create_monthly_units_visualizations(data):
    """Create visualizations for monthly units by category"""
    
    # 1. Grouped Bar Chart - Monthly Units by Category
    plt.figure(figsize=(16, 10))
    
    # Define colors for each category
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Create grouped bar chart
    x = np.arange(len(data.index))  # Month positions
    width = 0.1  # Width of bars
    
    for i, category in enumerate(data.columns):
        plt.bar(x + i * width, data[category], width, label=category, color=colors[i % len(colors)], alpha=0.8)
    
    plt.title('Monthly Units Sold by Item Category', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Units Sold', fontsize=12)
    plt.xticks(x + width * (len(data.columns) - 1) / 2, data.index)
    plt.legend(title='Item Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/monthly_units_by_category_grouped.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Stacked Bar Chart - Monthly Units by Category
    plt.figure(figsize=(14, 8))
    data.plot(kind='bar', stacked=True, figsize=(14, 8), color=colors[:len(data.columns)])
    plt.title('Monthly Units Sold by Item Category (Stacked)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Units Sold', fontsize=12)
    plt.legend(title='Item Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/monthly_units_by_category_stacked.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Line Chart - Monthly Units by Category
    plt.figure(figsize=(14, 8))
    for i, category in enumerate(data.columns):
        plt.plot(data.index, data[category], marker='o', linewidth=2, 
                label=category, color=colors[i % len(colors)], markersize=6)
    
    plt.title('Monthly Units Sold by Item Category (Line Chart)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Units Sold', fontsize=12)
    plt.xticks(data.index)
    plt.legend(title='Item Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/monthly_units_by_category_lines.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap - Monthly Units by Category
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.T, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Units Sold'})
    plt.title('Monthly Units Sold by Item Category (Heatmap)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Item Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/monthly_units_by_category_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ All monthly units visualizations saved")
    
    # 5. Summary statistics
    print("\n=== Summary Statistics ===")
    total_units_by_category = data.sum().sort_values(ascending=False)
    print("Total Units Sold by Category:")
    for category, units in total_units_by_category.items():
        print(f"  {category}: {units:,} units")
    
    avg_monthly_units = data.mean().sort_values(ascending=False)
    print("\nAverage Monthly Units by Category:")
    for category, units in avg_monthly_units.items():
        print(f"  {category}: {units:.1f} units/month")
    
    peak_month_by_category = data.idxmax()
    print("\nPeak Month by Category:")
    for category, month in peak_month_by_category.items():
        peak_units = data.loc[month, category]
        print(f"  {category}: Month {month} ({peak_units:,} units)")

def main():
    """Main function"""
    create_monthly_units_by_category()
    
    print("\n=== MONTHLY UNITS BY CATEGORY ANALYSIS COMPLETE ===")
    print("All visualizations saved to Byte_Datasets/figures/merchandise_sales/")
    print("Data saved to Byte_Datasets/monthly_units_by_category.csv")

if __name__ == "__main__":
    main()
