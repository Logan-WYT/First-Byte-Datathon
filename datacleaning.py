import pandas as pd

stadium_operations = "/Users/lelandgraves/Desktop/BOLT Datathon/BOLT UBC First Byte - Stadium Operations.xlsx"
df_stadium_operations = pd.read_excel(stadium_operations)

merchandise_sales = "/Users/lelandgraves/Desktop/BOLT Datathon/BOLT UBC First Byte - Merchandise Sales.xlsx"
df_merchandise_sales = pd.read_excel(merchandise_sales)

fanbase_engagement = "/Users/lelandgraves/Desktop/BOLT Datathon/BOLT UBC First Byte - Fanbase Engagement.xlsx"
df_fanbase_engagement = pd.read_excel(fanbase_engagement)

# Display the contents of each dataset
print("=" * 80)
print("STADIUM OPERATIONS DATA:")
print("=" * 80)
print(df_stadium_operations.head())
print(f"\nShape: {df_stadium_operations.shape}")
print(f"Columns: {list(df_stadium_operations.columns)}")

print("\n" + "=" * 80)
print("MERCHANDISE SALES DATA:")
print("=" * 80)
print(df_merchandise_sales.head())
print(f"\nShape: {df_merchandise_sales.shape}")
print(f"Columns: {list(df_merchandise_sales.columns)}")

print("\n" + "=" * 80)
print("FANBASE ENGAGEMENT DATA:")
print("=" * 80)
print(df_fanbase_engagement.head())
print(f"\nShape: {df_fanbase_engagement.shape}")
print(f"Columns: {list(df_fanbase_engagement.columns)}")

# Join merchandise_sales and fanbase_engagement by Member_ID
df_merged = pd.merge(
    df_merchandise_sales, 
    df_fanbase_engagement, 
    left_on='Member_ID', 
    right_on='Membership_ID', 
    how='inner'
)

print("\n" + "=" * 80)
print("MERGED DATA (Merchandise Sales + Fanbase Engagement):")
print("=" * 80)
print(df_merged.head())
print(f"\nShape: {df_merged.shape}")
print(f"Columns: {list(df_merged.columns)}")
print(f"\nMerge Info:")
print(f"  - Original Merchandise Sales records: {df_merchandise_sales.shape[0]}")
print(f"  - Original Fanbase Engagement records: {df_fanbase_engagement.shape[0]}")
print(f"  - Merged records: {df_merged.shape[0]}")

# Find data inconsistencies
print("\n" + "=" * 80)
print("DATA QUALITY CHECK - INCONSISTENCIES:")
print("=" * 80)

# 1. Check for age group inconsistencies
age_group_inconsistent = df_merged[df_merged['Customer_Age_Group'] != df_merged['Age_Group']]
print(f"\n1. AGE GROUP INCONSISTENCIES:")
print(f"   Records with mismatched age groups: {len(age_group_inconsistent)}")
if len(age_group_inconsistent) > 0:
    print(f"\n   Sample of inconsistent age groups:")
    print(age_group_inconsistent[['Member_ID', 'Customer_Age_Group', 'Age_Group']].head(10))
    print(f"\n   Unique combinations:")
    print(age_group_inconsistent.groupby(['Customer_Age_Group', 'Age_Group']).size().reset_index(name='Count'))

# 2. Check for region inconsistencies
region_inconsistent = df_merged[df_merged['Customer_Region_x'] != df_merged['Customer_Region_y']]
print(f"\n2. REGION INCONSISTENCIES:")
print(f"   Records with mismatched regions: {len(region_inconsistent)}")
if len(region_inconsistent) > 0:
    print(f"\n   Sample of inconsistent regions:")
    print(region_inconsistent[['Member_ID', 'Customer_Region_x', 'Customer_Region_y']].head(10))
    print(f"\n   Unique combinations:")
    print(region_inconsistent.groupby(['Customer_Region_x', 'Customer_Region_y']).size().reset_index(name='Count'))

# 3. Check for missing values
print(f"\n3. MISSING VALUES:")
missing_data = df_merged.isnull().sum()
missing_data = missing_data[missing_data > 0]
if len(missing_data) > 0:
    print(missing_data)
else:
    print("   No missing values found.")

# 4. Check for duplicate Member_IDs with different information
print(f"\n4. DUPLICATE MEMBER IDs WITH VARYING DATA:")
duplicate_members = df_merged[df_merged.duplicated(subset=['Member_ID'], keep=False)]
if len(duplicate_members) > 0:
    print(f"   Total records with duplicate Member_IDs: {len(duplicate_members)}")
    print(f"   Unique members with duplicates: {duplicate_members['Member_ID'].nunique()}")
    sample_member = duplicate_members['Member_ID'].iloc[0]
    print(f"\n   Example - Member {sample_member}:")
    print(duplicate_members[duplicate_members['Member_ID'] == sample_member][['Member_ID', 'Item_Name', 'Customer_Age_Group', 'Age_Group', 'Customer_Region_x', 'Customer_Region_y']].head())
else:
    print("   No duplicate Member_IDs found.")

# 5. Summary of data quality issues
print("\n" + "=" * 80)
print("SUMMARY OF DATA QUALITY ISSUES:")
print("=" * 80)
total_issues = len(age_group_inconsistent) + len(region_inconsistent)
print(f"Total records with inconsistencies: {total_issues} out of {len(df_merged)} ({total_issues/len(df_merged)*100:.2f}%)")
print(f"  - Age group mismatches: {len(age_group_inconsistent)}")
print(f"  - Region mismatches: {len(region_inconsistent)}")

# DATA RECONCILIATION
print("\n" + "=" * 80)
print("DATA RECONCILIATION:")
print("=" * 80)

# Create a cleaned version of the merged dataset
df_cleaned = df_merged.copy()

# 1. Fix Age Group - Use Fanbase Engagement as authority
print("\n1. RECONCILING AGE GROUPS:")
print(f"   Before: {df_cleaned['Customer_Age_Group'].value_counts().to_dict()}")
df_cleaned['Customer_Age_Group'] = df_cleaned['Age_Group']
print(f"   After: {df_cleaned['Customer_Age_Group'].value_counts().to_dict()}")
print("   ✓ All age groups now match fanbase engagement data")

# 2. Fix Regions - Use Fanbase Engagement as authority, map Canada = Domestic, others = International
print("\n2. RECONCILING REGIONS:")
print(f"   Countries in fanbase data: {df_cleaned['Customer_Region_y'].value_counts().to_dict()}")

# Create standardized region column
df_cleaned['Customer_Region_Standardized'] = df_cleaned['Customer_Region_y'].apply(
    lambda x: 'Domestic' if x == 'Canada' else 'International'
)

print(f"\n   Before (from merchandise): {df_cleaned['Customer_Region_x'].value_counts().to_dict()}")
print(f"   After (standardized): {df_cleaned['Customer_Region_Standardized'].value_counts().to_dict()}")
print("   ✓ Regions standardized: Canada = Domestic, All others = International")

# Replace the old region columns with the standardized one
df_cleaned = df_cleaned.drop(columns=['Customer_Region_x', 'Customer_Region_y'])
df_cleaned = df_cleaned.rename(columns={'Customer_Region_Standardized': 'Customer_Region'})

# 3. Remove duplicate ID column (Membership_ID is same as Member_ID)
df_cleaned = df_cleaned.drop(columns=['Membership_ID'])

# 4. Clean up Age_Group column (now redundant since we fixed Customer_Age_Group)
df_cleaned = df_cleaned.drop(columns=['Age_Group'])

# Display cleaned data
print("\n" + "=" * 80)
print("CLEANED DATASET:")
print("=" * 80)
print(df_cleaned.head(10))
print(f"\nShape: {df_cleaned.shape}")
print(f"Columns: {list(df_cleaned.columns)}")

# Verify no more inconsistencies
print("\n" + "=" * 80)
print("VERIFICATION - Post-Cleaning Check:")
print("=" * 80)
print(f"✓ Total records: {len(df_cleaned)}")
print(f"✓ Unique members: {df_cleaned['Member_ID'].nunique()}")
print(f"✓ Missing values in critical fields:")
missing_cleaned = df_cleaned[['Customer_Age_Group', 'Customer_Region']].isnull().sum()
print(missing_cleaned)
print(f"\n✓ Age group distribution:")
print(df_cleaned['Customer_Age_Group'].value_counts())
print(f"\n✓ Region distribution:")
print(df_cleaned['Customer_Region'].value_counts())