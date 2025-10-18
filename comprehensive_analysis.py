import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
import re

warnings.filterwarnings('ignore')

# Configuration
EST_SPEND_PER_GAME = 35.0
TOP_N_SOURCES = 5

def create_directories():
    """Create necessary directories for outputs"""
    directories = ['Byte_Datasets', 'Byte_Datasets/figures']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Created output directories")

def audit(df, name):
    """Audit dataframe for basic info and issues"""
    print(f"\n=== {name} Audit ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    return df

def save_fig(path, fig=None):
    """Save figure with proper formatting"""
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: {path}")

def clean_stadium_operations(df):
    """Clean Stadium Operations data"""
    print("\n=== Cleaning Stadium Operations ===")
    
    # Month: numeric 1-12 (coerce, drop invalid
    df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
    df = df.dropna(subset=['Month'])
    df = df[(df['Month'] >= 1) & (df['Month'] <= 12)]
    
    # Source: strip spaces, title-case
    df['Source'] = df['Source'].str.strip().str.title()
    
    # Revenue: strip $ and commas, convert to float
    df['Revenue'] = df['Revenue'].astype(str).str.replace('$', '').str.replace(',', '')
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    
    # Drop rows missing any essential columns
    df = df.dropna(subset=['Month', 'Source', 'Revenue'])
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    print(f"✓ Stadium Operations cleaned: {df.shape[0]} rows")
    return df

def clean_merchandise_sales(df):
    """Clean Merchandise Sales data"""
    print("\n=== Cleaning Merchandise Sales ===")
    
    # Trim strings
    string_cols = ['Product_ID', 'Barcode', 'Item_Category', 'Item_Name', 'Size', 
                   'Customer_Region', 'Channel', 'Member_ID']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Customer_Region: title-case
    df['Customer_Region'] = df['Customer_Region'].str.title()
    
    # Customer_Age_Group: uppercase, remove inner spaces
    df['Customer_Age_Group'] = df['Customer_Age_Group'].str.upper().str.replace(' ', '')
    
    # Dates: parse to datetime
    df['Selling_Date'] = pd.to_datetime(df['Selling_Date'], errors='coerce')
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
    
    # Unit_Price: strip $ and commas → float
    df['Unit_Price'] = df['Unit_Price'].astype(str).str.replace('$', '').str.replace(',', '')
    df['Unit_Price'] = pd.to_numeric(df['Unit_Price'], errors='coerce')
    
    # Essential columns for valid row
    essential_cols = ['Product_ID', 'Item_Category', 'Unit_Price', 'Selling_Date']
    df = df.dropna(subset=essential_cols)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    print(f"✓ Merchandise Sales cleaned: {df.shape[0]} rows")
    return df

def clean_fanbase_engagement(df):
    """Clean Fanbase Engagement data"""
    print("\n=== Cleaning Fanbase Engagement ===")
    
    # Age_Group: uppercase, remove inner spaces
    df['Age_Group'] = df['Age_Group'].str.upper().str.replace(' ', '')
    
    # Customer_Region: title-case
    df['Customer_Region'] = df['Customer_Region'].str.title()
    
    # Seasonal_Pass: map to boolean-like string
    seasonal_pass_map = {'TRUE': 'TRUE', 'YES': 'TRUE', 'Y': 'TRUE', '1': 'TRUE', 1: 'TRUE', True: 'TRUE'}
    df['Seasonal_Pass'] = df['Seasonal_Pass'].map(seasonal_pass_map).fillna('FALSE')
    
    # Games_Attended: numeric int (coerce → NaN → fill 0 → int)
    df['Games_Attended'] = pd.to_numeric(df['Games_Attended'], errors='coerce').fillna(0).astype(int)
    
    # Essential columns
    df = df.dropna(subset=['Membership_ID', 'Age_Group'])
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    print(f"✓ Fanbase Engagement cleaned: {df.shape[0]} rows")
    return df

def stadium_eda(stadium_df):
    """Stadium Operations EDA"""
    print("\n=== Stadium Operations EDA ===")
    
    # Monthly revenue
    monthly_rev = stadium_df.groupby('Month')['Revenue'].sum().reset_index()
    print("\nMonthly Revenue:")
    print(monthly_rev)
    monthly_rev.to_csv('Byte_Datasets/monthly_revenue.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_rev['Month'], monthly_rev['Revenue'], marker='o', linewidth=2, markersize=8)
    plt.title('Monthly Revenue Trend')
    plt.xlabel('Month')
    plt.ylabel('Revenue ($)')
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/monthly_revenue.png')
    plt.close()
    
    # Source revenue
    source_rev = stadium_df.groupby('Source')['Revenue'].sum().sort_values(ascending=False)
    print("\nRevenue by Source:")
    print(source_rev)
    source_rev.to_csv('Byte_Datasets/source_revenue.csv')
    
    plt.figure(figsize=(12, 8))
    source_rev.plot(kind='bar')
    plt.title('Revenue by Source')
    plt.xlabel('Source')
    plt.ylabel('Revenue ($)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/source_revenue.png')
    plt.close()
    
    # Monthly composition (stacked bar)
    monthly_composition = stadium_df.pivot_table(values='Revenue', index='Month', columns='Source', fill_value=0)
    print("\nMonthly Composition (first 5 rows):")
    print(monthly_composition.head())
    monthly_composition.to_csv('Byte_Datasets/monthly_composition.csv')
    
    plt.figure(figsize=(14, 8))
    monthly_composition.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Monthly Revenue Composition by Source')
    plt.xlabel('Month')
    plt.ylabel('Revenue ($)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/monthly_composition.png')
    plt.close()
    
    # Correlation analysis
    correlation_matrix = monthly_composition.corr()
    print("\nSource Correlation Matrix:")
    print(correlation_matrix)
    correlation_matrix.to_csv('Byte_Datasets/source_correlation.csv')
    
    return monthly_rev, source_rev, monthly_composition

def merchandise_eda(merch_df):
    """Merchandise Sales EDA"""
    print("\n=== Merchandise Sales EDA ===")
    
    # Category sales
    cat_sales = merch_df.groupby('Item_Category')['Unit_Price'].sum().sort_values(ascending=False)
    print("\nSales by Category:")
    print(cat_sales)
    cat_sales.to_csv('Byte_Datasets/category_sales.csv')
    
    plt.figure(figsize=(12, 8))
    cat_sales.plot(kind='bar')
    plt.title('Sales by Item Category')
    plt.xlabel('Category')
    plt.ylabel('Revenue ($)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/category_sales.png')
    plt.close()
    
    # Region sales
    region_sales = merch_df.groupby('Customer_Region')['Unit_Price'].sum()
    print("\nSales by Region:")
    print(region_sales)
    region_sales.to_csv('Byte_Datasets/region_sales.csv')
    
    plt.figure(figsize=(8, 6))
    region_sales.plot(kind='bar')
    plt.title('Sales by Customer Region')
    plt.xlabel('Region')
    plt.ylabel('Revenue ($)')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/region_sales.png')
    plt.close()
    
    # Promotion lift
    promo_sales = merch_df.groupby('Promotion')['Unit_Price'].sum()
    print("\nSales by Promotion:")
    print(promo_sales)
    
    if len(promo_sales) == 2:  # Both TRUE and FALSE present
        promo_lift = ((promo_sales[True] - promo_sales[False]) / promo_sales[False]) * 100
        print(f"Promotion Lift: {promo_lift:.2f}%")
    
    promo_sales.to_csv('Byte_Datasets/promotion_sales.csv')
    
    plt.figure(figsize=(8, 6))
    promo_sales.plot(kind='bar')
    plt.title('Sales by Promotion Status')
    plt.xlabel('Promotion')
    plt.ylabel('Revenue ($)')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/promotion_sales.png')
    plt.close()
    
    # Channel sales
    channel_sales = merch_df.groupby('Channel')['Unit_Price'].sum().sort_values(ascending=False)
    print("\nSales by Channel:")
    print(channel_sales)
    channel_sales.to_csv('Byte_Datasets/channel_sales.csv')
    
    plt.figure(figsize=(10, 6))
    channel_sales.plot(kind='bar')
    plt.title('Sales by Channel')
    plt.xlabel('Channel')
    plt.ylabel('Revenue ($)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/channel_sales.png')
    plt.close()
    
    # Age group sales
    age_sales = merch_df.groupby('Customer_Age_Group')['Unit_Price'].sum().sort_values(ascending=False)
    print("\nSales by Age Group:")
    print(age_sales)
    age_sales.to_csv('Byte_Datasets/age_sales.csv')
    
    plt.figure(figsize=(10, 6))
    age_sales.plot(kind='bar')
    plt.title('Sales by Customer Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Revenue ($)')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/age_sales.png')
    plt.close()
    
    return cat_sales, region_sales, promo_sales, channel_sales, age_sales

def fanbase_eda(fanbase_df):
    """Fanbase Engagement EDA"""
    print("\n=== Fanbase Engagement EDA ===")
    
    # Attendance histogram
    plt.figure(figsize=(10, 6))
    plt.hist(fanbase_df['Games_Attended'], bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Games Attended')
    plt.xlabel('Games Attended')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/attendance_histogram.png')
    plt.close()
    
    # Pass impact
    pass_impact = fanbase_df.groupby('Seasonal_Pass')['Games_Attended'].mean()
    print("\nAverage Games Attended by Seasonal Pass:")
    print(pass_impact)
    pass_impact.to_csv('Byte_Datasets/pass_impact.csv')
    
    plt.figure(figsize=(8, 6))
    pass_impact.plot(kind='bar')
    plt.title('Average Games Attended by Seasonal Pass Status')
    plt.xlabel('Seasonal Pass')
    plt.ylabel('Average Games Attended')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/pass_impact.png')
    plt.close()
    
    # Age attendance
    age_attendance = fanbase_df.groupby('Age_Group')['Games_Attended'].mean().sort_values(ascending=False)
    print("\nAverage Games Attended by Age Group:")
    print(age_attendance)
    age_attendance.to_csv('Byte_Datasets/age_attendance.csv')
    
    plt.figure(figsize=(10, 6))
    age_attendance.plot(kind='bar')
    plt.title('Average Games Attended by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Average Games Attended')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/age_attendance.png')
    plt.close()
    
    # Region members
    region_members = fanbase_df['Customer_Region'].value_counts()
    print("\nMembers by Region:")
    print(region_members)
    region_members.to_csv('Byte_Datasets/region_members.csv')
    
    plt.figure(figsize=(8, 6))
    region_members.plot(kind='bar')
    plt.title('Members by Region')
    plt.xlabel('Region')
    plt.ylabel('Number of Members')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    save_fig('Byte_Datasets/figures/region_members.png')
    plt.close()
    
    return pass_impact, age_attendance, region_members

def cross_pillar_overlay(merch_df, fanbase_df):
    """Create cross-pillar overlay analysis"""
    print("\n=== Cross-Pillar Overlay Analysis ===")
    
    # Get unified age groups
    merch_ages = set(merch_df['Customer_Age_Group'].unique())
    fanbase_ages = set(fanbase_df['Age_Group'].unique())
    unified_ages = sorted(merch_ages.union(fanbase_ages))
    
    # Merchandise revenue by age group
    merch_revenue = merch_df.groupby('Customer_Age_Group')['Unit_Price'].sum()
    
    # Estimated in-stadium spend
    est_instadium_spend = fanbase_df.groupby('Age_Group')['Games_Attended'].sum() * EST_SPEND_PER_GAME
    
    # Season pass count
    season_pass_count = fanbase_df[fanbase_df['Seasonal_Pass'] == 'TRUE'].groupby('Age_Group').size()
    
    # Create summary dataframe
    summary_data = []
    for age in unified_ages:
        merch_rev = merch_revenue.get(age, 0)
        stadium_spend = est_instadium_spend.get(age, 0)
        pass_count = season_pass_count.get(age, 0)
        summary_data.append({
            'Age_Group': age,
            'Merch_Revenue': merch_rev,
            'Est_InStadium_Spend': stadium_spend,
            'Season_Pass_Count': pass_count
        })
    
    summary_by_age = pd.DataFrame(summary_data).set_index('Age_Group')
    
    # Scale season pass count for visualization
    max_dollar = max(summary_by_age['Merch_Revenue'].max(), summary_by_age['Est_InStadium_Spend'].max())
    max_passes = summary_by_age['Season_Pass_Count'].max()
    scale_factor = max_dollar / max_passes if max_passes > 0 else 1
    
    summary_by_age['Season_Pass_Scaled'] = summary_by_age['Season_Pass_Count'] * scale_factor
    
    print("\nSummary by Age Group:")
    print(summary_by_age)
    summary_by_age.to_csv('Byte_Datasets/summary_by_age.csv')
    
    # Create overlay chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(summary_by_age))
    width = 0.25
    
    ax.bar(x - width, summary_by_age['Merch_Revenue'], width, label='Merchandise Revenue ($)', alpha=0.8)
    ax.bar(x, summary_by_age['Est_InStadium_Spend'], width, label='Est. In-Stadium Spend ($)', alpha=0.8)
    ax.bar(x + width, summary_by_age['Season_Pass_Scaled'], width, 
           label=f'Season Pass Count (×{scale_factor:.0f})', alpha=0.8)
    
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Revenue/Value ($)')
    ax.set_title('Cross-Pillar Analysis by Age Group')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_by_age.index)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_fig('Byte_Datasets/figures/overlay_by_age.png', fig)
    plt.close()
    
    return summary_by_age

def executive_tables(stadium_df, merch_df, fanbase_df):
    """Generate executive summary tables"""
    print("\n=== Executive Summary Tables ===")
    
    # Top stadium sources
    top_sources = stadium_df.groupby('Source')['Revenue'].sum().sort_values(ascending=False).head(TOP_N_SOURCES)
    print("\nTop Stadium Sources by Revenue:")
    print(top_sources)
    top_sources.to_csv('Byte_Datasets/top_sources.csv')
    
    # Top merchandise categories
    top_categories = merch_df.groupby('Item_Category')['Unit_Price'].sum().sort_values(ascending=False)
    print("\nTop Merchandise Categories by Revenue:")
    print(top_categories.head(10))
    top_categories.to_csv('Byte_Datasets/top_categories.csv')
    
    # Engagement KPIs
    avg_games_all = fanbase_df['Games_Attended'].mean()
    avg_games_pass = fanbase_df[fanbase_df['Seasonal_Pass'] == 'TRUE']['Games_Attended'].mean()
    pass_penetration = (fanbase_df['Seasonal_Pass'] == 'TRUE').mean() * 100
    members_domestic = (fanbase_df['Customer_Region'] == 'Domestic').sum()
    members_international = (fanbase_df['Customer_Region'] == 'International').sum()
    
    engagement_kpis = pd.DataFrame({
        'Metric': ['Avg Games (All)', 'Avg Games (Pass=TRUE)', 'Pass Penetration %', 
                  'Members Domestic', 'Members International'],
        'Value': [avg_games_all, avg_games_pass, pass_penetration, members_domestic, members_international]
    })
    
    print("\nEngagement KPIs:")
    print(engagement_kpis)
    engagement_kpis.to_csv('Byte_Datasets/engagement_kpis.csv')
    
    return top_sources, top_categories, engagement_kpis

def impact_model():
    """Simple impact model with scenario levers"""
    print("\n=== Impact Model ===")
    
    # Scenario levers
    scenarios = {
        'Base': {'attendance_delta': 0, 'concession_delta': 0, 'promo_lift': 0},
        'Optimistic': {'attendance_delta': 15, 'concession_delta': 10, 'promo_lift': 25},
        'Conservative': {'attendance_delta': 5, 'concession_delta': 3, 'promo_lift': 10}
    }
    
    # Base assumptions (these would come from actual data)
    base_annual_attendance = 100000  # Example
    base_concession_per_game = 25
    base_merch_revenue = 5000000
    
    results = []
    for scenario_name, params in scenarios.items():
        # Calculate incremental revenue
        attendance_impact = base_annual_attendance * (params['attendance_delta'] / 100) * base_concession_per_game
        concession_impact = base_annual_attendance * (params['concession_delta'] / 100) * base_concession_per_game
        merch_impact = base_merch_revenue * (params['promo_lift'] / 100)
        
        total_impact = attendance_impact + concession_impact + merch_impact
        
        results.append({
            'Scenario': scenario_name,
            'Attendance_Delta_%': params['attendance_delta'],
            'Concession_Delta_%': params['concession_delta'],
            'Promo_Lift_%': params['promo_lift'],
            'Stadium_Impact_$': attendance_impact + concession_impact,
            'Merch_Impact_$': merch_impact,
            'Total_Impact_$': total_impact
        })
    
    impact_results = pd.DataFrame(results)
    print("\nImpact Model Results:")
    print(impact_results)
    impact_results.to_csv('Byte_Datasets/impact_model.csv')
    
    return impact_results

def main():
    """Main execution function"""
    print("=== ByteData Comprehensive Analysis ===")
    
    # Create directories
    create_directories()
    
    # Check for input files
    input_files = ['Fanbase_Engagement.xlsx', 'Merchandise_Sales.xlsx', 'Stadium_Operations.xlsx']
    for file in input_files:
        if not os.path.exists(file):
            print(f"❌ Missing input file: {file}")
            return
        else:
            print(f"✓ Found input file: {file}")
    
    print(f"\nOutput directories: Byte_Datasets/, Byte_Datasets/figures/")
    
    # Load data
    print("\n=== Loading Data ===")
    stadium_df = pd.read_excel('Stadium_Operations.xlsx')
    merch_df = pd.read_excel('Merchandise_Sales.xlsx')
    fanbase_df = pd.read_excel('Fanbase_Engagement.xlsx')
    
    # Audit original data
    audit(stadium_df, "Stadium Operations (Original)")
    audit(merch_df, "Merchandise Sales (Original)")
    audit(fanbase_df, "Fanbase Engagement (Original)")
    
    # Clean data
    print("\n=== Data Cleaning ===")
    stadium_clean = clean_stadium_operations(stadium_df.copy())
    merch_clean = clean_merchandise_sales(merch_df.copy())
    fanbase_clean = clean_fanbase_engagement(fanbase_df.copy())
    
    # Save cleaned data
    stadium_clean.to_csv('Byte_Datasets/clean_stadium.csv', index=False)
    merch_clean.to_csv('Byte_Datasets/clean_merch.csv', index=False)
    fanbase_clean.to_csv('Byte_Datasets/clean_fanbase.csv', index=False)
    print("✓ Saved cleaned datasets")
    
    # EDA Analysis
    print("\n=== Exploratory Data Analysis ===")
    stadium_eda(stadium_clean)
    merchandise_eda(merch_clean)
    fanbase_eda(fanbase_clean)
    
    # Cross-pillar analysis
    print("\n=== Cross-Pillar Analysis ===")
    summary_by_age = cross_pillar_overlay(merch_clean, fanbase_clean)
    
    # Executive tables
    print("\n=== Executive Summary ===")
    executive_tables(stadium_clean, merch_clean, fanbase_clean)
    
    # Impact model
    print("\n=== Impact Modeling ===")
    impact_model()
    
    print("\n=== Analysis Complete ===")
    print("All outputs saved to Byte_Datasets/ directory")
    print("Figures saved to Byte_Datasets/figures/ directory")

if __name__ == "__main__":
    main()
