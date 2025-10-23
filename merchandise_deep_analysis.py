import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def merchandise_deep_analysis():
    """Comprehensive deep-dive analysis of Merchandise Sales"""
    print("=== Merchandise Sales Deep Analysis ===")
    
    # Load merchandise data
    merch_df = pd.read_csv('Byte_Datasets/clean_merch.csv')
    
    print(f"Loaded merchandise data: {merch_df.shape}")
    print(f"Date range: {merch_df['Selling_Date'].min()} to {merch_df['Selling_Date'].max()}")
    
    # Convert Selling_Date to datetime and extract month
    merch_df['Selling_Date'] = pd.to_datetime(merch_df['Selling_Date'])
    merch_df['Month'] = merch_df['Selling_Date'].dt.month
    
    # 1. Core Product–Demographic Matches: Age Group × Item Category
    analyze_age_category_combinations(merch_df)
    
    # 2. Region × Item Category Analysis
    analyze_region_category_combinations(merch_df)
    
    # 3. Channel × Promotion Status Analysis
    analyze_channel_promotion_combinations(merch_df)
    
    # 4. Month × Promotion Status/Channel Timing Analysis
    analyze_monthly_promotion_timing(merch_df)
    
    print("✓ All merchandise deep analysis completed!")

def analyze_age_category_combinations(df):
    """Analyze Age Group × Item Category combinations"""
    print("\n=== 1. Age Group × Item Category Analysis ===")
    
    # Create cross-tabulation
    age_category_crosstab = pd.crosstab(df['Customer_Age_Group'], df['Item_Category'], 
                                       values=df['Unit_Price'], aggfunc='sum').fillna(0)
    
    print("Age Group × Item Category Revenue Matrix:")
    print(age_category_crosstab)
    age_category_crosstab.to_csv('Byte_Datasets/age_category_revenue_matrix.csv')
    
    # Calculate percentage share of total sales per age group
    age_category_pct = age_category_crosstab.div(age_category_crosstab.sum(axis=1), axis=0) * 100
    print("\nPercentage Share of Total Sales per Age Group:")
    print(age_category_pct.round(2))
    age_category_pct.to_csv('Byte_Datasets/age_category_percentage_share.csv')
    
    # Create visualizations
    create_age_category_visualizations(age_category_crosstab, age_category_pct)

def create_age_category_visualizations(crosstab, pct_share):
    """Create visualizations for age group × category analysis"""
    
    # 1. Grouped Bar Chart - Revenue by Age Group and Category
    plt.figure(figsize=(16, 10))
    crosstab.plot(kind='bar', figsize=(16, 10), width=0.8)
    plt.title('Merchandise Revenue by Age Group and Item Category', fontsize=16, fontweight='bold')
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.legend(title='Item Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/age_category_grouped_bar.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap - Revenue Matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(crosstab, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Revenue ($)'})
    plt.title('Age Group × Item Category Revenue Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Item Category', fontsize=12)
    plt.ylabel('Age Group', fontsize=12)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/age_category_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Percentage Share Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pct_share, annot=True, fmt='.1f', cmap='Blues', 
                cbar_kws={'label': 'Percentage Share (%)'})
    plt.title('Age Group × Item Category - Percentage Share of Sales', fontsize=16, fontweight='bold')
    plt.xlabel('Item Category', fontsize=12)
    plt.ylabel('Age Group', fontsize=12)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/age_category_percentage_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Age × Category visualizations saved")

def analyze_region_category_combinations(df):
    """Analyze Region × Item Category combinations"""
    print("\n=== 2. Region × Item Category Analysis ===")
    
    # Create cross-tabulation
    region_category_crosstab = pd.crosstab(df['Customer_Region'], df['Item_Category'], 
                                          values=df['Unit_Price'], aggfunc='sum').fillna(0)
    
    print("Region × Item Category Revenue Matrix:")
    print(region_category_crosstab)
    region_category_crosstab.to_csv('Byte_Datasets/region_category_revenue_matrix.csv')
    
    # Calculate percentage of total category sales by region
    region_category_pct = region_category_crosstab.div(region_category_crosstab.sum(axis=0), axis=1) * 100
    print("\nPercentage of Total Category Sales by Region:")
    print(region_category_pct.round(2))
    region_category_pct.to_csv('Byte_Datasets/region_category_percentage_share.csv')
    
    # Create visualizations
    create_region_category_visualizations(region_category_crosstab, region_category_pct)

def create_region_category_visualizations(crosstab, pct_share):
    """Create visualizations for region × category analysis"""
    
    # 1. Stacked Bar Chart - Revenue by Region and Category
    plt.figure(figsize=(14, 8))
    crosstab.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Merchandise Revenue by Region and Item Category (Stacked)', fontsize=16, fontweight='bold')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.legend(title='Item Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/region_category_stacked_bar.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 100% Stacked Bar Chart - Percentage Distribution
    plt.figure(figsize=(14, 8))
    pct_share.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Merchandise Sales Distribution by Region and Category (100% Stacked)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Percentage of Category Sales (%)', fontsize=12)
    plt.legend(title='Item Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/region_category_100_stacked.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Region × Category visualizations saved")

def analyze_channel_promotion_combinations(df):
    """Analyze Channel × Promotion Status combinations"""
    print("\n=== 3. Channel × Promotion Status Analysis ===")
    
    # Create cross-tabulation
    channel_promo_crosstab = pd.crosstab(df['Channel'], df['Promotion'], 
                                        values=df['Unit_Price'], aggfunc='sum').fillna(0)
    
    print("Channel × Promotion Revenue Matrix:")
    print(channel_promo_crosstab)
    channel_promo_crosstab.to_csv('Byte_Datasets/channel_promotion_revenue_matrix.csv')
    
    # Calculate promotion lift per channel
    promo_lift = {}
    for channel in channel_promo_crosstab.index:
        non_promo_sales = channel_promo_crosstab.loc[channel, False]
        promo_sales = channel_promo_crosstab.loc[channel, True]
        if non_promo_sales > 0:
            lift = ((promo_sales / non_promo_sales) - 1) * 100
            promo_lift[channel] = lift
        else:
            promo_lift[channel] = 0
    
    promo_lift_df = pd.DataFrame(list(promo_lift.items()), columns=['Channel', 'Promotion_Lift_%'])
    print("\nPromotion Lift by Channel:")
    print(promo_lift_df)
    promo_lift_df.to_csv('Byte_Datasets/channel_promotion_lift.csv', index=False)
    
    # Create visualizations
    create_channel_promotion_visualizations(channel_promo_crosstab, promo_lift_df)

def create_channel_promotion_visualizations(crosstab, promo_lift_df):
    """Create visualizations for channel × promotion analysis"""
    
    # 1. Clustered Bar Chart - Revenue by Channel and Promotion
    plt.figure(figsize=(12, 8))
    crosstab.plot(kind='bar', figsize=(12, 8), width=0.8)
    plt.title('Merchandise Revenue by Channel and Promotion Status', fontsize=16, fontweight='bold')
    plt.xlabel('Channel', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.legend(title='Promotion Status', labels=['No Promotion', 'Promotion'])
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/channel_promotion_clustered_bar.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Promotion Lift Bar Chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(promo_lift_df['Channel'], promo_lift_df['Promotion_Lift_%'], 
                   color=['green' if x > 0 else 'red' for x in promo_lift_df['Promotion_Lift_%']])
    plt.title('Promotion Lift by Channel', fontsize=16, fontweight='bold')
    plt.xlabel('Channel', fontsize=12)
    plt.ylabel('Promotion Lift (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/channel_promotion_lift.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Channel × Promotion visualizations saved")

def analyze_monthly_promotion_timing(df):
    """Analyze Month × Promotion Status/Channel timing"""
    print("\n=== 4. Monthly Promotion Timing Analysis ===")
    
    # Month × Promotion analysis
    month_promo_crosstab = pd.crosstab(df['Month'], df['Promotion'], 
                                      values=df['Unit_Price'], aggfunc='sum').fillna(0)
    
    # Calculate promotion share percentage for each month
    month_promo_pct = {}
    for month in month_promo_crosstab.index:
        non_promo_revenue = month_promo_crosstab.loc[month, False]
        promo_revenue = month_promo_crosstab.loc[month, True]
        total_revenue = non_promo_revenue + promo_revenue
        if total_revenue > 0:
            promo_share = (promo_revenue / total_revenue) * 100
            month_promo_pct[month] = promo_share
        else:
            month_promo_pct[month] = 0
    
    promo_share_df = pd.DataFrame(list(month_promo_pct.items()), columns=['Month', 'Promo_Share_%'])
    promo_share_df = promo_share_df.sort_values('Month')
    
    print("Monthly Promotion Share:")
    print(promo_share_df)
    promo_share_df.to_csv('Byte_Datasets/monthly_promotion_share.csv', index=False)
    
    # Month × Channel analysis
    month_channel_crosstab = pd.crosstab(df['Month'], df['Channel'], 
                                        values=df['Unit_Price'], aggfunc='sum').fillna(0)
    
    print("\nMonthly Revenue by Channel:")
    print(month_channel_crosstab)
    month_channel_crosstab.to_csv('Byte_Datasets/monthly_channel_revenue.csv')
    
    # Channel performance by month (ranking)
    channel_ranking = month_channel_crosstab.rank(axis=1, ascending=False, method='min')
    print("\nChannel Ranking by Month (1=Best):")
    print(channel_ranking)
    channel_ranking.to_csv('Byte_Datasets/monthly_channel_ranking.csv')
    
    # Month × Channel × Promotion analysis
    month_channel_promo = df.groupby(['Month', 'Channel', 'Promotion'])['Unit_Price'].sum().unstack(fill_value=0)
    
    # Calculate promotion share by month and channel
    month_channel_promo_pct = {}
    for month in month_channel_promo.index.get_level_values(0).unique():
        for channel in month_channel_promo.index.get_level_values(1).unique():
            try:
                non_promo = month_channel_promo.loc[(month, channel), False]
                promo = month_channel_promo.loc[(month, channel), True]
                total = non_promo + promo
                if total > 0:
                    pct = (promo / total) * 100
                    month_channel_promo_pct[(month, channel)] = pct
                else:
                    month_channel_promo_pct[(month, channel)] = 0
            except KeyError:
                month_channel_promo_pct[(month, channel)] = 0
    
    month_channel_promo_pct_df = pd.DataFrame(list(month_channel_promo_pct.items()), 
                                            columns=['Month_Channel', 'Promo_Share_%'])
    month_channel_promo_pct_df[['Month', 'Channel']] = pd.DataFrame(
        month_channel_promo_pct_df['Month_Channel'].tolist(), index=month_channel_promo_pct_df.index)
    month_channel_promo_pct_df = month_channel_promo_pct_df.pivot(index='Month', columns='Channel', values='Promo_Share_%')
    
    print("\nMonthly Promotion Share by Channel:")
    print(month_channel_promo_pct_df)
    month_channel_promo_pct_df.to_csv('Byte_Datasets/monthly_channel_promotion_share.csv')
    
    # Create visualizations
    create_monthly_timing_visualizations(month_promo_crosstab, promo_share_df, 
                                        month_channel_crosstab, month_channel_promo_pct_df)

def create_monthly_timing_visualizations(month_promo_crosstab, promo_share_df, 
                                        month_channel_crosstab, month_channel_promo_pct):
    """Create visualizations for monthly timing analysis"""
    
    # 1. Monthly Promotion Share Line Chart
    plt.figure(figsize=(14, 8))
    plt.plot(promo_share_df['Month'], promo_share_df['Promo_Share_%'], 
             marker='o', linewidth=3, markersize=8, color='purple')
    plt.title('Monthly Promotion Share - When Promotions Matter Most', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Promotion Share (%)', fontsize=12)
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Add value labels
    for i, row in promo_share_df.iterrows():
        plt.text(row['Month'], row['Promo_Share_%'] + 2, f"{row['Promo_Share_%']:.1f}%", 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/monthly_promotion_share_timing.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monthly Revenue by Channel Line Chart
    plt.figure(figsize=(14, 8))
    for channel in month_channel_crosstab.columns:
        plt.plot(month_channel_crosstab.index, month_channel_crosstab[channel], 
                marker='o', linewidth=2, label=channel, markersize=6)
    
    plt.title('Monthly Revenue by Channel - Channel Performance Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.xticks(range(1, 13))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/monthly_channel_performance.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Monthly Channel Ranking Heatmap
    plt.figure(figsize=(12, 8))
    channel_ranking = month_channel_crosstab.rank(axis=1, ascending=False, method='min')
    sns.heatmap(channel_ranking, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Rank (1=Best)'})
    plt.title('Monthly Channel Performance Ranking', fontsize=16, fontweight='bold')
    plt.xlabel('Channel', fontsize=12)
    plt.ylabel('Month', fontsize=12)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/monthly_channel_ranking_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Monthly Revenue by Channel - Simple Bar Chart
    plt.figure(figsize=(14, 8))
    month_channel_crosstab.plot(kind='bar', figsize=(14, 8), width=0.8)
    plt.title('Monthly Revenue by Channel', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Channel')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Byte_Datasets/figures/merchandise_sales/monthly_channel_revenue_bars.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Monthly timing visualizations saved")

def main():
    """Main function to run all merchandise deep analysis"""
    merchandise_deep_analysis()
    
    print("\n=== MERCHANDISE DEEP ANALYSIS COMPLETE ===")
    print("All analysis files saved to Byte_Datasets/")
    print("All visualizations saved to Byte_Datasets/figures/merchandise_sales/")

if __name__ == "__main__":
    main()
