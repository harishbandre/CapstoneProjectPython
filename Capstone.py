# 1. importing

# Core data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical testing (used later for the hypothesis testing)
from scipy import stats

# Fixed seed so the 'random' dataset is identical every time this notebook runs.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 2. Generate the Dataset

# Number of 'clean' base transaction we will start from, before injecting messy data.

N_BASE_ROWS = 5000

# Define the building blocks of our fictional retail business
categories = ['Electronics', 'Apparel', 'Home & Kitchen', 'Beauty', 'Sports', 'Toys']
regions = ['North', 'South', 'East', 'West']

# A small product catalog per category, with realistic base price range.
# Using a dict keeps category and product price ranges logically linked.
product_catalog = {
    'Electronics': {'products': ['Wireless Earbuds', 'Bluetooth Speaker', 'Smartwatch', 'Laptop Stand'], 'price_range': (15, 250)},
    'Apparel': {'products': ['Cotton T-Shirt', 'Denim Jeans', 'Running Shoes', 'Winter Jacket'], 'price_range': (10, 120)},
    'Home & Kitchen': {'products': ['Non-Stick Pan', 'Blender', 'Throw Pillow', 'LED Lamp'], 'price_range': (8, 90)},
    'Beauty': {'products': ['Face Serum', 'Lip Balm Set', 'Hair Dryer', 'Makeup Brush Kit'], 'price_range': (5, 60)},
    'Sports': {'products': ['Yoga Mat', 'Resistance Bands', 'Water Bottle', 'Dumbbell Set'], 'price_range': (10, 150)},
    'Toys': {'products': ['Building Blocks', 'Puzzle 1000pc', 'RC Car', 'Plush Toy'], 'price_range': (6, 80)},
}

# Generate a pool of 800 customer IDs. Not every customer orders the same number of
# times - we use np.random.choice with replacement so some customers naturally
# appear far more often than others, which is exactly what real repeat-purchase
# behaviour looks like (and what makes RFM meaningful later)
customer_ids = [f'CUST{str(i).zfill(4)}' for i in range(1, 801)]

# Build the base transaction table row by row
rows = []

date_range = pd.date_range(start = '2025-01-01', end = '2025-12-31',freq ='D')

for _ in range(N_BASE_ROWS):
  category = np.random.choice(categories)
  product = np.random.choice(product_catalog[category]['products'])
  low, high = product_catalog[category]['price_range']
  unit_price = round(np.random.uniform(low, high), 2)

  # Slight seasonal boost: more order land in Nov/Dec (holiday shopping)
  # simulated by givign later dates a higher sampling weight
  weights = np.linspace(1, 2.2, len(date_range))
  order_date = np.random.choice(date_range, p = weights / weights.sum())

  rows.append({
      'OrderDate': order_date,
      'CustomerID': np.random.choice(customer_ids),
      'Category': category,
      'Product': product,
      'Quantity': np.random.randint(1, 6),
      'UnitPrice': unit_price,
      'Region': np.random.choice(regions)
  })

df = pd.DataFrame(rows)

# Now Inject realistic mess on top of the clean base data.

# 1) Duplicate ~2% of rows exactly
dupes = df.sample(frac = 0.02, random_state = RANDOM_STATE)
df = pd.concat([df, dupes], ignore_index = True)

# 2) Introduce missing values: ~3% missing Quantity, ~4% missing region
missing_qty_idx = df.sample(frac = 0.03, random_state=RANDOM_STATE).index
df.loc[missing_qty_idx, 'Quantity'] = np.nan

missing_region_idx = df.sample(frac = 0.04, random_state = RANDOM_STATE + 1).index
df.loc[missing_region_idx, 'Region'] = np.nan

# 3) Inconsistent casing in ~10% of category values
messy_case_idx = df.sample(frac = 0.1, random_state = RANDOM_STATE + 2).index
df.loc[messy_case_idx, 'Category'] = df.loc[messy_case_idx, 'Category'].str.lower()

# 4) A few extereme outliers 'bulk orders'
outlier_idx = df.sample(n = 8, random_state = RANDOM_STATE+ 3).index

df.loc[outlier_idx, 'Quantity'] = np.random.randint(80, 150, size = len(outlier_idx))

# 3. Data cleaning & preprocessing
# Before any analysis, we always inspect the data first, then clean methodically. We never silently fix something we have not measure first

# Inspect data quality before fixing

print(f'Data types: \n {df.dtypes}')
print(f'Missing values per column: \n{df.isna().sum()}')
print(f'Exact duplicate rows: \n{df.duplicated().sum()}')
print(f'Unique category values (note the inconsistent casing): \n{sorted(df['Category'].unique())}')

# Remove exact duplicate rows
# We remove duplicates BEFORE Imputing missing values, so duplicated missing
# Value does not get counted

before = len(df)
df = df.drop_duplicates().reset_index(drop = True)
print(f'Removed {before - len(df)} duplicate rows. New shape: {df.shape}')

# Standardize text fields
# Title-case the category column so 'electronics' and 'Electronics' are treaded
# as the same group

df['Category'] = df['Category'].str.title()
print(f'Categories after standardizing: {sorted(df['Category'].unique())}')

# Handle missing Values

median_qty = df['Quantity'].median()
df['Quantity'] = df['Quantity'].fillna(median_qty)
df['Region'] = df['Region'].fillna('Unknown')
print(f'Missing values remainig: {df.isna().sum().sum()} total')

cap = df['Quantity'].quantile(0.99)
n_capped = (df['Quantity'] > cap).sum()
df['Quantity'] = np.where(df['Quantity'] > cap, cap, df['Quantity'])
print(f'Capped {n_capped} exterme Quantity values at the 99th percentile ({cap}) units')

df['Quantity'] = df['Quantity'].astype(int)

df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# Revenue is the core metrics the rest of the notebook will build on
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Convenience date parts used repatedly in later section

df['Month'] = df['OrderDate'].dt.to_period('M').astype(str)
df['DayOfWeek'] = df['OrderDate'].dt.day_name()
df['IsWeekend'] = df['OrderDate'].dt.dayofweek >=5 # Saturday = 5, # Sunday = 6

print(f'Final dataset shape {df.shape}')
df.head()

# 4. Exploratory Data Analysis
# Revenue Distribution

fig, axes = plt.subplots(1, 2, figsize = (14, 5))

sns.histplot(df['Revenue'], bins = 40, kde = True, ax = axes[0], color = 'steelblue')
axes[0].set_title('Distribution of Order Revenue')
axes[0].set_xlabel('Revenue ($)')

sns.boxplot(x = df['Revenue'], ax = axes[1], color = 'lightcoral')
axes[1].set_title('Order Revenue - Boxplot')
axes[1].set_xlabel('Revenue ($)')

# Revenue by category

revenue_by_category = (
    df.groupby('Category')['Revenue'].sum().sort_values(ascending = False)
)

plt.figure(figsize = (9, 5))
sns.barplot(x = revenue_by_category.values, y = revenue_by_category.index, palette = 'viridis')
plt.title('Total revenue by Category')
plt.xlabel('Revenue ($)')
plt.ylabel('Category')
revenue_by_category

# Revenue by region

revenue_by_region = (
    df.groupby('Region')['Revenue']
    .sum()
    .sort_values(ascending = False)
)

plt.figure(figsize = (8, 5))
sns.barplot(x = revenue_by_region.index, y = revenue_by_region.values, palette = 'magma')

plt.title('Total Revenue by Region')
plt.ylabel('Revenue ($)')
plt.show()

revenue_by_region

# 5. Time-Based Trend Analysis
# Monthly revenue trend
# This is the single most common chart a business stakeholder asks for:
# "is revenue going up or down?"

monthly_revenue = df.groupby('Month')['Revenue'].sum().sort_index()

plt.figure(figsize = (12, 5))
monthly_revenue.plot(kind = 'line', marker = 'o', color = 'darkgreen')

plt.title('Monthly Revenue Trend')
plt.ylabel('Revenue')
plt.xlabel('Month')
plt.xticks(rotation = 45)
plt.grid(True, alpha = 0.3)

monthly_revenue

# Weekday vs weekend pattern
# Ordered explicitly (Mon -> Sun) so the chart reads naturally instead of
# alphabetically, which is the default pandas would show

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
revenue_by_day = df.groupby('DayOfWeek')['Revenue'].sum().reindex(day_order)

plt.figure(figsize = (10, 5))

sns.barplot(x = revenue_by_day.index, y = revenue_by_day.values, palette = 'coolwarm')

plt.title('Total Revenue by Day of week')
plt.ylabel('Revenue ($)')
plt.xticks(rotation = 30)
plt.show()

# 6. Customer Segmentation (RFM Analysis)

# RFM Scores each customer on three dimensions
# Recency - days since their last order (lower = more recently active)
# Frequency - number of distinct orders placed
# Monetary - total revenue contributed
# We score each dimension 1-4 using quartiles, combine into a single RFM segment label and use that to identify our most valuable customers vs those at risk of churning

# Reference data = one day after the last order in the dataset, so 'days since last order' is
# always a positive number, even for someone who ordered on the last day

snapshot_date = df['OrderDate'].max() + pd.Timedelta(days = 1)

rfm = df.groupby('CustomerID').agg(
    Recency = ('OrderDate', lambda x: (snapshot_date - x.max()).days),
    Frequency = ('OrderDate', 'count'),
    Monetary = ('Revenue', 'sum')
).reset_index()

rfm.head()

# Score each metric 1-4 using quartiles (4 = best)
# Recency is reversed (qcut labels 4, 3, 2, 1) because a SMALL recency (recent order is GOOD)
# the opposite direction for Frequency/Monetary where bigger is better

rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels = [4, 3, 2, 1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method = 'first'), 4, labels = [1, 2, 3 , 4]).astype(int)
rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels = [1, 2, 3, 4]).astype(int)

rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# Translate the numeric score into a plain-language segement label -
# this is the part a non-technical person actually wants to see

def label_segment(score):
  if score >= 10:
    return 'Champions'
  elif score >=8:
    return 'Loyal Customers'
  elif score >= 6:
    return 'Potential Loyalists'
  elif score >=4:
    return 'At risk'
  else:
    return 'Lost/Inactive'

rfm['Segment'] = rfm['RFM_score'].apply(label_segment)

# Visualize how many customers - and how much revenue - fall into each segment

segment_summary = rfm.groupby('Segment').agg(
    Customers = ('CustomerID', 'count'),
    TotalRevenue = ('Monetary', 'sum')
).sort_values('TotalRevenue', ascending = False)

segment_summary

fig, axes = plt.subplots(1, 2, figsize = (14, 5))
sns.barplot(x = segment_summary.index, y = segment_summary['Customers'], ax = axes[0], palette = 'Set2')
axes[0].set_title('Number of Customers per Segment')
axes[0].tick_params(axis = 'x', rotation = 30)

sns.barplot(x = segment_summary.index , y = segment_summary['TotalRevenue'], ax = axes[1], palette = 'Set2')
axes[1].set_title('Total Revenue per Segment')
axes[1].set_ylabel('Revenue ($)')
axes[1].tick_params(axis = 'x', rotation = 30)

# 7. Product Performance

# Top 10 Products by revenue - useful for inventory/marketing priortization

top_products = (
    df.groupby('Product')['Revenue']
    .sum()
    .sort_values(ascending = False)
    .head(10)
)
top_products

plt.figure(figsize = (9, 6))
sns.barplot(x = top_products.values, y = top_products.index, palette = 'crest')
plt.title('Top 10 Products by Revenue')
plt.xlabel('Revenue ($)')
plt.show()

# 8. Correlation & Hypothesis Testing

# Correlation heatmap

numeric_cols = ['Quantity', 'UnitPrice', 'Revenue']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize = (6, 5))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm')

plt.title('Correlation Matrix')
plt.show()

# 9. Business Insights & Recommendations

# Pulling together key numbers computed above into one readable summary block
# In a real job, this is the paragraph that goes into the email to your manager.

top_category = revenue_by_category.idxmax()
top_region = revenue_by_region.idxmax()
best_month = monthly_revenue.idxmax()

champions_count = (rfm['Segment'] == 'Champions').sum()
champions_revenue_share = rfm.loc[rfm['Segment'] == 'Champions', 'Monetary'].sum() / rfm['Monetary']


print('KEY INSIGHTS')
print('=' * 50)

print(f'1. {top_category} is the top-performing category by revenue')
print(f'2. The {top_region} region generates the most revenue')
print(f'3. {best_month} was the strongest month - consistent with holiday-season lift')

print(f'4. {champions_count} customers (Champions segment) drive' f'{champions_revenue_share} of total revenue')

