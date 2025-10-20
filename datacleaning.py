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

# VISUALIZATIONS
import matplotlib.pyplot as plt

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS:")
print("=" * 80)

# 1. Attendance by Age Group
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Get attendance data by age group
attendance_by_age = df_cleaned.groupby('Customer_Age_Group')['Games_Attended'].agg(['sum', 'mean', 'count'])
attendance_by_age = attendance_by_age.sort_values('sum', ascending=False)

# Bar chart for total attendance
ax1.bar(attendance_by_age.index, attendance_by_age['sum'], color='#1f77b4', edgecolor='black', linewidth=1.2)
ax1.set_title('Total Games Attended by Age Group', fontsize=14, fontweight='bold')
ax1.set_xlabel('Age Group', fontsize=12)
ax1.set_ylabel('Total Games Attended', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, value) in enumerate(attendance_by_age['sum'].items()):
    ax1.text(i, value, f'{int(value):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Average attendance per person by age group
avg_attendance = df_cleaned.groupby('Customer_Age_Group')['Games_Attended'].mean().sort_index()
ax2.bar(avg_attendance.index, avg_attendance.values, color='#ff7f0e', edgecolor='black', linewidth=1.2)
ax2.set_title('Average Games Attended per Person by Age Group', fontsize=14, fontweight='bold')
ax2.set_xlabel('Age Group', fontsize=12)
ax2.set_ylabel('Average Games Attended', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, value) in enumerate(avg_attendance.items()):
    ax2.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/lelandgraves/Desktop/BOLT Datathon/attendance_by_age_group.png', dpi=300, bbox_inches='tight')
print("✓ Saved: attendance_by_age_group.png")
plt.close()

# 2. Sales by Age Group
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Get sales data by age group
sales_by_age = df_cleaned.groupby('Customer_Age_Group')['Unit_Price'].agg(['sum', 'mean', 'count'])
sales_by_age = sales_by_age.sort_values('sum', ascending=False)

# Bar chart for total sales
ax1.bar(sales_by_age.index, sales_by_age['sum'], color='#2ca02c', edgecolor='black', linewidth=1.2)
ax1.set_title('Total Merchandise Sales by Age Group', fontsize=14, fontweight='bold')
ax1.set_xlabel('Age Group', fontsize=12)
ax1.set_ylabel('Total Sales ($)', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, value) in enumerate(sales_by_age['sum'].items()):
    ax1.text(i, value, f'${value:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Average spending per transaction by age group
avg_spending = df_cleaned.groupby('Customer_Age_Group')['Unit_Price'].mean().sort_index()
ax2.bar(avg_spending.index, avg_spending.values, color='#d62728', edgecolor='black', linewidth=1.2)
ax2.set_title('Average Spending per Transaction by Age Group', fontsize=14, fontweight='bold')
ax2.set_xlabel('Age Group', fontsize=12)
ax2.set_ylabel('Average Transaction ($)', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, value) in enumerate(avg_spending.items()):
    ax2.text(i, value, f'${value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/lelandgraves/Desktop/BOLT Datathon/sales_by_age_group.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sales_by_age_group.png")
plt.close()

print("\n" + "=" * 80)
print("SUMMARY STATISTICS:")
print("=" * 80)
print("\nAttendance by Age Group:")
print(attendance_by_age)
print("\nSales by Age Group:")
print(sales_by_age)

# ADDITIONAL VISUALIZATIONS - STADIUM OPERATIONS & DELIVERY TIMES
print("\n" + "=" * 80)
print("CREATING ADDITIONAL VISUALIZATIONS:")
print("=" * 80)

# 1. Revenue by Source over Time (Line Chart)
fig, ax = plt.subplots(figsize=(14, 7))

# Pivot data to have months as rows and sources as columns
revenue_by_source = df_stadium_operations.pivot(index='Month', columns='Source', values='Revenue')

# Plot each source as a separate line
for source in revenue_by_source.columns:
    ax.plot(revenue_by_source.index, revenue_by_source[source], marker='o', linewidth=2, label=source)

ax.set_title('Revenue by Source Over Time', fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Revenue ($)', fontsize=12)
ax.legend(title='Source', fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/lelandgraves/Desktop/BOLT Datathon/revenue_by_source_over_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved: revenue_by_source_over_time.png")
plt.close()

# 2. Total Profit by Month (Line Chart)
fig, ax = plt.subplots(figsize=(14, 7))

# Calculate total profit (sum of all revenues, negatives are costs) per month
monthly_profit = df_stadium_operations.groupby('Month')['Revenue'].sum()

ax.plot(monthly_profit.index, monthly_profit.values, marker='o', linewidth=3, color='#2ca02c', markersize=8)
ax.fill_between(monthly_profit.index, 0, monthly_profit.values, alpha=0.3, color='#2ca02c')

# Add value labels
for x, y in zip(monthly_profit.index, monthly_profit.values):
    ax.text(x, y, f'${y:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title('Total Monthly Profit (Revenue - Costs)', fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Profit ($)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig('/Users/lelandgraves/Desktop/BOLT Datathon/monthly_profit.png', dpi=300, bbox_inches='tight')
print("✓ Saved: monthly_profit.png")
plt.close()

# 3. Delivery Time: Domestic vs International (Bar Chart)
fig, ax = plt.subplots(figsize=(12, 7))

# Calculate delivery time (only for records with both dates)
df_delivery = df_cleaned[df_cleaned['Arrival_Date'].notna()].copy()
df_delivery['Delivery_Time'] = (pd.to_datetime(df_delivery['Arrival_Date']) - pd.to_datetime(df_delivery['Selling_Date'])).dt.days

# Calculate average delivery time by region
avg_delivery = df_delivery.groupby('Customer_Region')['Delivery_Time'].agg(['mean', 'median', 'count'])
avg_delivery = avg_delivery.sort_values('mean', ascending=False)

# Create bar chart
bars = ax.bar(avg_delivery.index, avg_delivery['mean'], color=['#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=1.5)

# Add value labels
for i, (idx, row) in enumerate(avg_delivery.iterrows()):
    ax.text(i, row['mean'], f"{row['mean']:.1f} days\n(n={int(row['count'])})", 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('Average Delivery Time: Domestic vs International', fontsize=16, fontweight='bold')
ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('Average Delivery Time (Days)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/lelandgraves/Desktop/BOLT Datathon/delivery_time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: delivery_time_comparison.png")
plt.close()

print("\n" + "=" * 80)
print("DELIVERY TIME STATISTICS:")
print("=" * 80)
print(avg_delivery)

# CUSTOMER TOTAL SPENDING BY DELIVERY TIME
print("\n" + "=" * 80)
print("CUSTOMER TOTAL SPENDING ANALYSIS BY DELIVERY TIME:")
print("=" * 80)

# Calculate delivery time for each customer (average if they have multiple orders)
customer_delivery = df_delivery.groupby('Member_ID').agg({
    'Delivery_Time': 'mean',  # Average delivery time for customers with multiple orders
    'Unit_Price': 'sum'  # Total spending over the season
}).reset_index()
customer_delivery.columns = ['Member_ID', 'Avg_Delivery_Time', 'Total_Season_Spending']

# Round delivery time to nearest day for categorization
customer_delivery['Delivery_Time_Category'] = customer_delivery['Avg_Delivery_Time'].round().astype(int)

# Calculate average total spending per customer for each delivery time category
spending_by_delivery_cat = customer_delivery.groupby('Delivery_Time_Category').agg({
    'Total_Season_Spending': ['mean', 'median', 'sum', 'count']
}).reset_index()
spending_by_delivery_cat.columns = ['Delivery_Days', 'Avg_Total_Spending', 'Median_Spending', 'Total_Revenue', 'Customer_Count']

# Filter to show reasonable delivery times (1-15 days for clarity)
spending_by_delivery_cat = spending_by_delivery_cat[spending_by_delivery_cat['Delivery_Days'] <= 15]

print("\nSpending Statistics by Delivery Time:")
print(spending_by_delivery_cat)

# PRODUCT DELIVERY TIME ANALYSIS
print("\n" + "=" * 80)
print("AVERAGE DELIVERY TIME BY PRODUCT:")
print("=" * 80)

# Calculate average delivery time for each product
product_delivery = df_delivery.groupby('Item_Name').agg({
    'Delivery_Time': ['mean', 'median', 'count']
}).reset_index()
product_delivery.columns = ['Product', 'Avg_Delivery_Time', 'Median_Delivery_Time', 'Order_Count']

# Sort by average delivery time
product_delivery = product_delivery.sort_values('Avg_Delivery_Time', ascending=False)

print("\nProduct Delivery Statistics:")
print(product_delivery)

# Create bar chart for products
fig, ax = plt.subplots(figsize=(14, 10))

# Create horizontal bar chart for better readability
bars = ax.barh(product_delivery['Product'], product_delivery['Avg_Delivery_Time'], 
               edgecolor='black', linewidth=1.2)

# Color bars based on delivery time (fast vs slow)
colors = ['#2ca02c' if x <= 7 else '#e74c3c' for x in product_delivery['Avg_Delivery_Time']]
for bar, color in zip(bars, colors):
    bar.set_color(color)

# Add value labels
for i, (idx, row) in enumerate(product_delivery.iterrows()):
    ax.text(row['Avg_Delivery_Time'], i, 
            f" {row['Avg_Delivery_Time']:.1f} days (n={int(row['Order_Count'])})", 
            va='center', fontsize=9, fontweight='bold')

# Add vertical line at 7 days to mark the fast/slow threshold
ax.axvline(x=7, color='black', linestyle='--', linewidth=2, alpha=0.5, label='7-day threshold')

ax.set_title('Average Delivery Time by Product', fontsize=16, fontweight='bold')
ax.set_xlabel('Average Delivery Time (Days)', fontsize=12)
ax.set_ylabel('Product Name', fontsize=12)
ax.grid(axis='x', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('/Users/lelandgraves/Desktop/BOLT Datathon/delivery_time_by_product.png', dpi=300, bbox_inches='tight')
print("✓ Saved: delivery_time_by_product.png")
plt.close()

# Create distribution chart
fig, ax = plt.subplots(figsize=(14, 7))

# Create bar chart
bars = ax.bar(spending_by_delivery_cat['Delivery_Days'], spending_by_delivery_cat['Avg_Total_Spending'], 
              color='#3498db', edgecolor='black', linewidth=1.2, alpha=0.8)

# Color bars differently for fast (<=7) vs slow (>7) delivery
for i, row in spending_by_delivery_cat.iterrows():
    if row['Delivery_Days'] <= 7:
        bars[i].set_color('#2ca02c')  # Green for fast delivery
    else:
        bars[i].set_color('#e74c3c')  # Red for slow delivery

# Add value labels on top of bars
for i, row in spending_by_delivery_cat.iterrows():
    if row['Customer_Count'] >= 100:  # Only show labels for significant sample sizes
        ax.text(row['Delivery_Days'], row['Avg_Total_Spending'], 
                f"${row['Avg_Total_Spending']:.0f}\n(n={int(row['Customer_Count'])})", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add vertical line at 7 days to mark the fast/slow threshold
ax.axvline(x=7.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='7-day threshold')

ax.set_title('Average Total Season Spending per Customer by Delivery Time', fontsize=16, fontweight='bold')
ax.set_xlabel('Delivery Time (Days)', fontsize=12)
ax.set_ylabel('Average Total Spending per Customer Over Season ($)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.legend()

# Set x-axis to show all integer days
ax.set_xticks(range(1, 16))

plt.tight_layout()
plt.savefig('/Users/lelandgraves/Desktop/BOLT Datathon/total_spending_by_delivery_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved: total_spending_by_delivery_time.png")
plt.close()

print("\n✓ All visualizations created successfully!")
