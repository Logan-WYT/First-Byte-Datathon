import pandas as pd
import numpy as np

def fix_age_group_inconsistencies(df, dataset_name):
    """Fix age group inconsistencies across datasets"""
    print(f"\n=== Fixing Age Group Inconsistencies for {dataset_name} ===")
    
    # Standardize age groups to consistent format
    age_group_mapping = {
        # Handle different dash types and standardize ranges
        '18-25': '18-25',
        '18–25': '18-25',  # en-dash to regular dash
        '26-35': '26-35',
        '26-40': '26-40',  # Keep this as is since it's a different range
        '26–40': '26-40',  # en-dash to regular dash
        '41-60': '41-60',
        '41–60': '41-60',  # en-dash to regular dash
        '60+': '60+',
        '<18': '<18'
    }
    
    if 'Age_Group' in df.columns:
        df['Age_Group'] = df['Age_Group'].str.strip().str.replace(' ', '')
        df['Age_Group'] = df['Age_Group'].map(age_group_mapping).fillna(df['Age_Group'])
        print(f"✓ Standardized Age_Group column")
    
    if 'Customer_Age_Group' in df.columns:
        df['Customer_Age_Group'] = df['Customer_Age_Group'].str.strip().str.replace(' ', '')
        df['Customer_Age_Group'] = df['Customer_Age_Group'].map(age_group_mapping).fillna(df['Customer_Age_Group'])
        print(f"✓ Standardized Customer_Age_Group column")
    
    return df

def fix_region_inconsistencies(df, dataset_name):
    """Fix region inconsistencies across datasets"""
    print(f"\n=== Fixing Region Inconsistencies for {dataset_name} ===")
    
    # Map cluster regions to Domestic/International
    region_mapping = {
        # Domestic (Canada)
        'Canada': 'Domestic',
        'canada': 'Domestic',
        'CANADA': 'Domestic',
        
        # International (all others)
        'Usa': 'International',
        'USA': 'International',
        'usa': 'International',
        'India': 'International',
        'india': 'International',
        'INDIA': 'International',
        'Japan': 'International',
        'japan': 'International',
        'JAPAN': 'International',
        'China': 'International',
        'china': 'International',
        'CHINA': 'International',
        'South Korea': 'International',
        'south korea': 'International',
        'SOUTH KOREA': 'International',
        'Uk': 'International',
        'UK': 'International',
        'uk': 'International',
        
        # Keep existing Domestic/International as is
        'Domestic': 'Domestic',
        'International': 'International',
        'domestic': 'Domestic',
        'international': 'International'
    }
    
    if 'Customer_Region' in df.columns:
        df['Customer_Region'] = df['Customer_Region'].str.strip()
        df['Customer_Region'] = df['Customer_Region'].map(region_mapping).fillna(df['Customer_Region'])
        print(f"✓ Standardized Customer_Region column")
        print(f"  Region distribution: {df['Customer_Region'].value_counts().to_dict()}")
    
    return df

def create_unified_age_groups():
    """Create a unified age group system for cross-dataset analysis"""
    # Define standard age groups that work across all datasets
    unified_age_groups = ['<18', '18-25', '26-35', '36-45', '46-60', '60+']
    return unified_age_groups

def main():
    """Fix all data inconsistencies"""
    print("=== Fixing Data Inconsistencies ===")
    
    # Load the cleaned datasets
    fanbase_df = pd.read_csv('Byte_Datasets/clean_fanbase.csv')
    merch_df = pd.read_csv('Byte_Datasets/clean_merch.csv')
    stadium_df = pd.read_csv('Byte_Datasets/clean_stadium.csv')
    
    print(f"Loaded datasets:")
    print(f"  Fanbase: {fanbase_df.shape}")
    print(f"  Merchandise: {merch_df.shape}")
    print(f"  Stadium: {stadium_df.shape}")
    
    # Show current inconsistencies
    print(f"\n=== Current Inconsistencies ===")
    print(f"Fanbase Age Groups: {sorted(fanbase_df['Age_Group'].unique())}")
    print(f"Fanbase Regions: {sorted(fanbase_df['Customer_Region'].unique())}")
    print(f"Merchandise Age Groups: {sorted(merch_df['Customer_Age_Group'].unique())}")
    print(f"Merchandise Regions: {sorted(merch_df['Customer_Region'].unique())}")
    
    # Fix inconsistencies
    fanbase_fixed = fix_age_group_inconsistencies(fanbase_df.copy(), "Fanbase")
    fanbase_fixed = fix_region_inconsistencies(fanbase_fixed, "Fanbase")
    
    merch_fixed = fix_age_group_inconsistencies(merch_df.copy(), "Merchandise")
    merch_fixed = fix_region_inconsistencies(merch_fixed, "Merchandise")
    
    # Show fixed inconsistencies
    print(f"\n=== After Fixing Inconsistencies ===")
    print(f"Fanbase Age Groups: {sorted(fanbase_fixed['Age_Group'].unique())}")
    print(f"Fanbase Regions: {sorted(fanbase_fixed['Customer_Region'].unique())}")
    print(f"Merchandise Age Groups: {sorted(merch_fixed['Customer_Age_Group'].unique())}")
    print(f"Merchandise Regions: {sorted(merch_fixed['Customer_Region'].unique())}")
    
    # Save fixed datasets
    fanbase_fixed.to_csv('Byte_Datasets/fixed_fanbase.csv', index=False)
    merch_fixed.to_csv('Byte_Datasets/fixed_merch.csv', index=False)
    stadium_df.to_csv('Byte_Datasets/fixed_stadium.csv', index=False)
    
    print(f"\n✓ Saved fixed datasets:")
    print(f"  fixed_fanbase.csv")
    print(f"  fixed_merch.csv") 
    print(f"  fixed_stadium.csv")
    
    # Create cross-dataset analysis with consistent age groups
    print(f"\n=== Cross-Dataset Analysis with Consistent Data ===")
    
    # Get unified age groups across both datasets
    fanbase_ages = set(fanbase_fixed['Age_Group'].unique())
    merch_ages = set(merch_fixed['Customer_Age_Group'].unique())
    unified_ages = sorted(fanbase_ages.union(merch_ages))
    
    print(f"Unified age groups: {unified_ages}")
    
    # Create summary by age group with consistent data
    summary_data = []
    for age in unified_ages:
        # Fanbase data
        fanbase_count = len(fanbase_fixed[fanbase_fixed['Age_Group'] == age])
        fanbase_avg_games = fanbase_fixed[fanbase_fixed['Age_Group'] == age]['Games_Attended'].mean()
        fanbase_domestic = len(fanbase_fixed[(fanbase_fixed['Age_Group'] == age) & 
                                           (fanbase_fixed['Customer_Region'] == 'Domestic')])
        fanbase_international = len(fanbase_fixed[(fanbase_fixed['Age_Group'] == age) & 
                                                (fanbase_fixed['Customer_Region'] == 'International')])
        
        # Merchandise data
        merch_revenue = merch_fixed[merch_fixed['Customer_Age_Group'] == age]['Unit_Price'].sum()
        merch_domestic = merch_fixed[(merch_fixed['Customer_Age_Group'] == age) & 
                                   (merch_fixed['Customer_Region'] == 'Domestic')]['Unit_Price'].sum()
        merch_international = merch_fixed[(merch_fixed['Customer_Age_Group'] == age) & 
                                        (merch_fixed['Customer_Region'] == 'International')]['Unit_Price'].sum()
        
        summary_data.append({
            'Age_Group': age,
            'Fanbase_Count': fanbase_count,
            'Avg_Games_Attended': fanbase_avg_games,
            'Fanbase_Domestic': fanbase_domestic,
            'Fanbase_International': fanbase_international,
            'Merch_Revenue_Total': merch_revenue,
            'Merch_Revenue_Domestic': merch_domestic,
            'Merch_Revenue_International': merch_international
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(f"\nConsistent Cross-Dataset Summary:")
    print(summary_df)
    summary_df.to_csv('Byte_Datasets/consistent_summary.csv', index=False)
    
    print(f"\n✓ Saved consistent_summary.csv")
    print(f"\n=== Data Inconsistencies Fixed Successfully! ===")

if __name__ == "__main__":
    main()
