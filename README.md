# ByteData Analysis - First Byte Datathon

## Overview
Comprehensive data analysis of stadium operations, merchandise sales, and fanbase engagement data for Vancouver City FC.

## Repository Structure
```
ByteData/
├── clean.py                          # Basic data cleaning script
├── comprehensive_analysis.py         # Full analysis pipeline
├── Fanbase_Engagement.xlsx          # Original fanbase data
├── Merchandise_Sales.xlsx           # Original merchandise data
├── Stadium_Operations.xlsx          # Original stadium operations data
├── Byte_Datasets/                   # Analysis outputs
│   ├── figures/                     # All generated charts
│   ├── clean_*.csv                  # Cleaned datasets
│   ├── *.csv                        # Analysis results
│   └── summary_by_age.csv          # Cross-pillar analysis
└── README.md                        # This file
```

## Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

### Run Analysis
```bash
python comprehensive_analysis.py
```

## Key Insights

### 🏟️ Stadium Operations
- **Top Revenue Sources:** Lower Bowl ($24.7M), Food ($19.9M), Advertising ($5.1M)
- **Monthly Trends:** Peak in February ($3.9M), lowest in January (-$1.9M)
- **Cost Centers:** Staff (-$39.6M), Maintenance (-$7.9M)

### 🛍️ Merchandise Sales
- **Top Categories:** Jersey ($4.1M), Hoodie ($1.0M), Youth Jersey ($598K)
- **Age Demographics:** 18-25 leads with $2.8M in sales
- **Channels:** Online dominates ($5.2M vs $1.3M Team Store)
- **Promotion Impact:** -43.68% lift (promotions underperforming)

### 👥 Fanbase Engagement
- **Season Pass Impact:** Pass holders attend 22.5 games vs 4.5 for non-pass holders
- **Pass Penetration:** 6.78% of members have season passes
- **Geographic Distribution:** 90% Canada, 7% USA, 3% International
- **Age Engagement:** 26-40 age group most engaged (5.77 avg games)

## Data Cleaning Features

### Stadium Operations
- ✅ Month validation (1-12)
- ✅ Source title-casing
- ✅ Revenue cleaning ($ and comma removal)
- ✅ Duplicate removal

### Merchandise Sales
- ✅ String trimming and formatting
- ✅ Date parsing (Selling_Date, Arrival_Date)
- ✅ Age group standardization
- ✅ Price cleaning and validation

### Fanbase Engagement
- ✅ Age group standardization
- ✅ Boolean mapping for Seasonal_Pass
- ✅ Games_Attended numeric conversion
- ✅ Region standardization

## Analysis Components

1. **Data Cleaning** - Robust preprocessing with validation
2. **EDA Analysis** - Comprehensive exploratory data analysis
3. **Cross-Pillar Analysis** - Unified age group analysis across datasets
4. **Executive Summary** - Key performance indicators and top metrics
5. **Impact Modeling** - Scenario planning with revenue projections

## Output Files

### Cleaned Data
- `clean_stadium.csv` - Cleaned stadium operations
- `clean_merch.csv` - Cleaned merchandise sales
- `clean_fanbase.csv` - Cleaned fanbase engagement

### Analysis Results
- `summary_by_age.csv` - Cross-pillar analysis by age group
- `top_sources.csv` - Top revenue sources
- `top_categories.csv` - Top merchandise categories
- `engagement_kpis.csv` - Key engagement metrics
- `impact_model.csv` - Scenario planning results

### Visualizations (PNG)
- Monthly revenue trends
- Revenue by source/category/region
- Attendance distribution
- Cross-pillar overlay charts
- And more...

## Business Impact

The analysis reveals significant opportunities:
- **Promotion Optimization:** Current promotions show negative lift (-43.68%)
- **Season Pass Growth:** Only 6.78% penetration with 5x higher engagement
- **Age Targeting:** 18-25 demographic drives highest merchandise sales
- **Channel Strategy:** Online sales dominate (80% of revenue)

## Technical Features

- Modular, robust data cleaning functions
- Comprehensive error handling
- Automated directory creation
- High-quality visualizations (300 DPI)
- Executive-ready summary tables
- Scenario-based impact modeling

## Author
Created for First Byte Datathon - Vancouver City FC Data Analysis
