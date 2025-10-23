# seasonal_pass_purchase_analysis.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Config / paths
# -------------------------
MERCH_FP = 'Byte_Datasets/clean_merch.csv'
FAN_FP   = 'Byte_Datasets/clean_fanbase.csv'
FIG_DIR  = 'Byte_Datasets/figures/merchandise_sales'
OUT_DIR  = 'Byte_Datasets'

# -------------------------
# Utils
# -------------------------
def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)

def to_bool_pass(x):
    """Robust mapping of pass flag to boolean (handles TRUE/FALSE, Yes/No, 1/0, strings)."""
    s = str(x).strip().upper()
    return s in {'TRUE','T','YES','Y','1'}

def safe_pct_lift(new, base, eps=1e-9):
    return ( (new / (base + eps)) - 1.0 ) * 100.0

# -------------------------
# Core analysis
# -------------------------
def merge_and_analyze_seasonal_pass_purchases():
    """
    Merge merchandise and fanbase on true member IDs and compute PER-MEMBER stats
    to compare seasonal pass holders vs non-pass members in real units.
    """
    print("=== Seasonal Pass vs Purchase Behavior (Member-level) ===")
    ensure_dirs()

    # Load
    merch_df = pd.read_csv(MERCH_FP)
    fanbase_df = pd.read_csv(FAN_FP)

    print(f"Merchandise data shape: {merch_df.shape}")
    print(f"Fanbase data shape:     {fanbase_df.shape}")

    # Normalize pass flag in fanbase
    fan = fanbase_df.copy()
    if 'Seasonal_Pass' not in fan.columns:
        raise KeyError("Fanbase file missing 'Seasonal_Pass' column.")
    fan['Seasonal_Pass'] = fan['Seasonal_Pass'].apply(to_bool_pass)

    # Keep 1 row/member in fanbase (if any dupes)
    if 'Membership_ID' not in fan.columns:
        raise KeyError("Fanbase file missing 'Membership_ID' column.")
    fan = fan.sort_values('Membership_ID').drop_duplicates('Membership_ID', keep='first')

    # Ensure required cols exist in merch
    for c in ['Member_ID','Unit_Price','Item_Category']:
        if c not in merch_df.columns:
            raise KeyError(f"Merchandise file missing '{c}' column.")

    # Inner join on IDs (ONLY overlapping members)
    use_cols_fan = ['Membership_ID','Seasonal_Pass','Games_Attended','Age_Group','Customer_Region']
    for c in use_cols_fan:
        if c not in fan.columns:
            raise KeyError(f"Fanbase file missing '{c}' column.")
    merged = merch_df.merge(
        fan[use_cols_fan],
        left_on='Member_ID', right_on='Membership_ID', how='inner',
        suffixes=('_merch','_fan')
    )

    # Basic overlap printout
    merch_members = merch_df['Member_ID'].nunique()
    fan_members   = fan['Membership_ID'].nunique()
    common_members = merged['Member_ID'].nunique()
    print(f"Merged rows: {merged.shape[0]:,} | Members (merch): {merch_members:,} | "
          f"Members (fan): {fan_members:,} | Common: {common_members:,} "
          f"({common_members / max(1, merch_members) * 100:.1f}% of merch members)")

    # PER-MEMBER aggregation (key requirement)
    member_stats = (merged
        .groupby(['Member_ID','Seasonal_Pass'], as_index=False)
        .agg(
            Purchase_Count = ('Unit_Price','count'),
            Total_Spent    = ('Unit_Price','sum'),
            Avg_Purchase   = ('Unit_Price','mean'),
            Games_Attended = ('Games_Attended','max'),
            Age_Group      = ('Age_Group','first'),
            Region         = ('Customer_Region_fan','first')  # from fanbase (post-merge)
        )
    )

    # Save core artifact
    member_stats.to_csv(os.path.join(OUT_DIR, 'seasonal_pass_member_stats.csv'), index=False)

    # Group means (no normalization)
    means = (member_stats
             .groupby('Seasonal_Pass')[['Purchase_Count','Total_Spent','Avg_Purchase','Games_Attended']]
             .mean().round(2))
    print("\nPer-member means by Seasonal_Pass (False=no pass, True=pass):")
    print(means)

    # Lift vs non-pass (if both groups exist)
    if set(means.index) == {False, True}:
        lift = ((means.loc[True] - means.loc[False]) / means.loc[False] * 100.0).round(1)
        lift.to_csv(os.path.join(OUT_DIR,'seasonal_pass_lift_vs_nopass.csv'))
        print("\nLift for pass-holders vs non-pass (%):")
        print(lift)

    # Category analysis (revenue share by pass)
    cat_rev = (merged.groupby(['Seasonal_Pass','Item_Category'])['Unit_Price']
               .sum().reset_index())
    cat_rev['Share_%'] = (cat_rev.groupby('Seasonal_Pass')['Unit_Price']
                          .transform(lambda s: 100 * s / s.sum()))
    cat_rev.to_csv(os.path.join(OUT_DIR,'seasonal_pass_category_share.csv'), index=False)

    # Make visuals
    create_seasonal_pass_visualizations(member_stats, merged)

    return merged, member_stats

def create_seasonal_pass_visualizations(member_stats, transactions):
    """Member-ID–based visuals only (no normalized proxies)."""
    ensure_dirs()
    member_stats = member_stats.copy()
    member_stats['Seasonal_Pass'] = member_stats['Seasonal_Pass'].astype(bool)

    # 1) Mean Total Spent per Member (with SE bars)
    grp = member_stats.groupby('Seasonal_Pass')['Total_Spent']
    means = grp.mean()
    stds  = grp.std().fillna(0.0)
    ns    = grp.count().clip(lower=1)

    figpath = os.path.join(FIG_DIR, 'pass_mean_total_spent.png')
    plt.figure(figsize=(7,5))
    x = np.array([0,1])
    y = means.reindex([False, True]).values if set(means.index)=={False,True} else means.values
    e = (stds / np.sqrt(ns)).reindex([False, True]).values if set(stds.index)=={False,True} else (stds / np.sqrt(ns)).values
    labels = ['No Pass','Pass'] if len(y)==2 else [str(idx) for idx in means.index]

    plt.bar(x, y, yerr=e, capsize=6)
    plt.xticks(x, labels)
    plt.ylabel("Mean Total Spent per Member ($)")
    plt.title("Merchandise Spending per Member by Seasonal Pass (with SE)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Distribution of Total Spent per Member
    figpath = os.path.join(FIG_DIR, 'pass_spend_distribution.png')
    plt.figure(figsize=(9,5))
    for flag, lab, color in [(False,'No Pass','tab:blue'), (True,'Pass','tab:orange')]:
        if flag in set(member_stats['Seasonal_Pass'].unique()):
            data = member_stats.loc[member_stats['Seasonal_Pass']==flag, 'Total_Spent']
            data = data[data > 0]
            if len(data) > 0:
                plt.hist(data, bins=30, alpha=0.55, label=lab, color=color)
    plt.xlabel("Total Spent per Member ($)")
    plt.ylabel("Member Count")
    plt.title("Distribution of Member Spending (by Seasonal Pass)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()

    # 3) Games Attended vs Total Spent (scatter)
    figpath = os.path.join(FIG_DIR, 'pass_games_vs_spend_scatter.png')
    plt.figure(figsize=(8,6))
    colors = {False:'tab:blue', True:'tab:orange'}
    for flag in [False, True]:
        if flag in set(member_stats['Seasonal_Pass'].unique()):
            df = member_stats[member_stats['Seasonal_Pass']==flag]
            plt.scatter(df['Games_Attended'], df['Total_Spent'], alpha=0.6,
                        label=('Pass' if flag else 'No Pass'), s=18, c=colors[flag])
    plt.xlabel("Games Attended (per Member)")
    plt.ylabel("Total Spent on Merchandise ($)")
    plt.title("Games Attended vs Merchandise Spend (by Seasonal Pass)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()

    # 4) Category Mix by Pass (share of revenue)
    figpath = os.path.join(FIG_DIR, 'pass_category_mix.png')
    cat_rev = (transactions.groupby(['Seasonal_Pass','Item_Category'])['Unit_Price']
               .sum().reset_index())
    cat_rev['Share_%'] = cat_rev.groupby('Seasonal_Pass')['Unit_Price']\
                                .transform(lambda s: 100*s/s.sum())
    cats = sorted(cat_rev['Item_Category'].unique())
    x = np.arange(len(cats))
    width = 0.38

    def series_for(flag):
        s = (cat_rev[cat_rev['Seasonal_Pass']==flag]
             .set_index('Item_Category')
             .reindex(cats)['Share_%']).fillna(0.0)
        return s

    no_pass = series_for(False) if False in set(cat_rev['Seasonal_Pass']) else pd.Series(0, index=cats, dtype=float)
    yes_pass = series_for(True) if True in set(cat_rev['Seasonal_Pass']) else pd.Series(0, index=cats, dtype=float)

    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, no_pass.values, width=width, label='No Pass')
    plt.bar(x + width/2, yes_pass.values, width=width, label='Pass')
    plt.xticks(x, cats, rotation=15)
    plt.ylabel("Category Share of Revenue (%)")
    plt.title("Category Mix by Seasonal Pass Group")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Replaced seasonal pass visuals with member-ID based charts.")

def calculate_purchase_lift():
    """
    Compute lift for pass-holders vs non-pass using member-level aggregates.
    Saves a compact CSV for your slide.
    """
    print("\n=== Purchase Lift (Member-level) ===")
    merch_df = pd.read_csv(MERCH_FP)
    fanbase_df = pd.read_csv(FAN_FP)

    fanbase_df['Seasonal_Pass'] = fanbase_df['Seasonal_Pass'].apply(to_bool_pass)

    merged = merch_df.merge(
        fanbase_df[['Membership_ID','Seasonal_Pass']],
        left_on='Member_ID', right_on='Membership_ID', how='inner',
        suffixes=('_merch','_fan')
    )

    member_stats = (merged
        .groupby(['Member_ID','Seasonal_Pass'], as_index=False)
        .agg(
            Purchase_Count=('Unit_Price','count'),
            Total_Spent   =('Unit_Price','sum'),
            Avg_Purchase  =('Unit_Price','mean')
        )
    )

    means = member_stats.groupby('Seasonal_Pass')[['Purchase_Count','Total_Spent','Avg_Purchase']].mean()

    if set(means.index) == {False, True}:
        base = means.loc[False]
        lift = ((means.loc[True] - base) / base * 100.0).rename('Lift_%').round(1)
        summary = pd.concat([base.rename('No_Pass_Mean'), means.loc[True].rename('Pass_Mean'), lift], axis=1)
    else:
        summary = means.copy()

    print(summary.round(2))
    summary.to_csv(os.path.join(OUT_DIR, 'seasonal_pass_lift_analysis.csv'))

# -------------------------
# Entrypoint
# -------------------------
def main():
    merged_df, member_stats = merge_and_analyze_seasonal_pass_purchases()
    calculate_purchase_lift()
    print("\n=== COMPLETE ===")
    print(f"Tables saved in: {OUT_DIR}/")
    print(f"Figures saved in: {FIG_DIR}/")
    print(f"Merged dataset rows: {merged_df.shape[0]:,} | Members compared: {member_stats['Member_ID'].nunique():,}")

if __name__ == '__main__':
    main()

