import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style
sns.set(style="whitegrid")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean_customer_data(filepath):
    print("Loading and Cleaning Customer Data...")
    df = pd.read_csv(filepath)
    
    # 1. Remove duplicates
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_len - len(df)} duplicates.")
    
    # 2. Fix date formats
    date_cols = ['Order Date', 'Signup Date', 'Last Active Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        
    # 3. Remove negative/invalid values
    # Order Amount should be positive
    invalid_orders = df[df['Order Amount'] < 0]
    df = df[df['Order Amount'] >= 0]
    print(f"Removed {len(invalid_orders)} rows with negative Order Amount.")
    
    # 4. Handle missing values (if any)
    # In our mock data, we might not have many, but let's check
    df.dropna(subset=['Customer ID', 'Order Date'], inplace=True)
    
    # 5. Standardize categorical fields
    if 'Product Category' in df.columns:
        df['Product Category'] = df['Product Category'].str.title()
        print("Standardized 'Product Category' column.")
    
    return df

def analyze_customer_behavior(df):
    print("\n--- Part A: Customer Behavior Analysis ---")
    
    # A. Most Active Customers
    orders_per_cust = df.groupby('Customer ID').size().reset_index(name='OrderCount')
    top_active = orders_per_cust.sort_values('OrderCount', ascending=False).head(5)
    print("Top 5 Most Active Customers (by Order Count):")
    print(top_active)
    
    # High vs Low Value Segments
    revenue_per_cust = df.groupby('Customer ID')['Order Amount'].sum().reset_index(name='TotalRevenue')
    # Simple segmentation: High Value > Median
    median_rev = revenue_per_cust['TotalRevenue'].median()
    revenue_per_cust['Segment'] = revenue_per_cust['TotalRevenue'].apply(lambda x: 'High Value' if x > median_rev else 'Low Value')
    
    print(f"Median Revenue per Customer: ${median_rev:.2f}")
    print("Customer Counts by Segment:")
    print(revenue_per_cust['Segment'].value_counts())
    
    # Revenue by Segment
    rev_by_segment = revenue_per_cust.groupby('Segment')['TotalRevenue'].sum()
    print("Total Revenue by Segment:")
    print(rev_by_segment)
    
    # B. Buying Frequency
    # Calculate time between purchases for each customer
    df_sorted = df.sort_values(['Customer ID', 'Order Date'])
    df_sorted['PrevOrderDate'] = df_sorted.groupby('Customer ID')['Order Date'].shift(1)
    df_sorted['DaysSincePrev'] = (df_sorted['Order Date'] - df_sorted['PrevOrderDate']).dt.days
    
    avg_days_between = df_sorted['DaysSincePrev'].mean()
    print(f"Average Days Between Purchases: {avg_days_between:.2f}")
    
    # Monthly Order Frequency
    df['OrderMonth'] = df['Order Date'].dt.to_period('M')
    monthly_freq = df.groupby('OrderMonth').size()
    
    # Weekly Order Frequency
    df['OrderWeek'] = df['Order Date'].dt.to_period('W')
    weekly_freq = df.groupby('OrderWeek').size()
    print(f"Average Weekly Orders: {weekly_freq.mean():.2f}")

    # C. Customer Retention Analysis
    # Retention Rate: Customers who bought in Month X and also in Month X+1
    # Simplified: % of customers who made more than 1 order
    repeat_buyers = orders_per_cust[orders_per_cust['OrderCount'] > 1].shape[0]
    total_cust = orders_per_cust.shape[0]
    retention_rate = (repeat_buyers / total_cust) * 100
    print(f"Retention Rate (Repeat Buyers): {retention_rate:.2f}%")
    
    # Churn Rate: Customers who haven't ordered in the last 90 days
    # Assuming 'today' is the max date in the dataset
    max_date = df['Order Date'].max()
    churn_threshold = max_date - pd.Timedelta(days=90)
    
    # Get last order date for each customer
    last_order_dates = df.groupby('Customer ID')['Order Date'].max()
    churned_customers = last_order_dates[last_order_dates < churn_threshold].count()
    churn_rate = (churned_customers / total_cust) * 100
    print(f"Churn Rate (Inactive > 90 days): {churn_rate:.2f}%")

    
    # D. Revenue Contribution
    top_10_rev = revenue_per_cust.sort_values('TotalRevenue', ascending=False).head(10)
    print("Top 10 Revenue Contributing Customers:")
    print(top_10_rev)
    
    # Pareto Analysis
    revenue_per_cust = revenue_per_cust.sort_values('TotalRevenue', ascending=False)
    revenue_per_cust['CumRevenue'] = revenue_per_cust['TotalRevenue'].cumsum()
    revenue_per_cust['CumRevPerc'] = 100 * revenue_per_cust['CumRevenue'] / revenue_per_cust['TotalRevenue'].sum()
    revenue_per_cust['CumCustPerc'] = 100 * (revenue_per_cust.reset_index().index + 1) / len(revenue_per_cust)
    
    # Visualizations
    
    # 1. Bar Chart: Revenue by Customer (Top 20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Customer ID', y='TotalRevenue', data=top_10_rev, palette='viridis')
    plt.title('Top 10 Customers by Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/revenue_by_customer.png")
    plt.close()
    
    # 2. Line Chart: Order Frequency Over Time
    plt.figure(figsize=(12, 6))
    monthly_freq.plot(kind='line', marker='o')
    plt.title('Monthly Order Frequency')
    plt.ylabel('Number of Orders')
    plt.xlabel('Month')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/order_frequency_over_time.png")
    plt.close()
    
    # 3. Pie Chart: Customer Segments
    plt.figure(figsize=(8, 8))
    revenue_per_cust['Segment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    plt.title('Customer Segments (High vs Low Value)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/customer_segments.png")
    plt.close()
    
    # 4. Heatmap: Retention Matrix (Cohort Analysis)
    # Cohort based on Signup Month
    df['CohortMonth'] = df['Signup Date'].dt.to_period('M')
    df['CohortIndex'] = (df['OrderMonth'] - df['CohortMonth']).apply(lambda x: x.n)
    
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['Customer ID'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='Customer ID')
    # Calculate retention as percentage of cohort size (index 0)
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='YlGnBu')
    plt.title('Customer Retention Heatmap (Cohorts)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/retention_heatmap.png")
    plt.close()

    # Save metrics to file for easy reading
    with open('metrics.txt', 'w') as f:
        f.write(f"Average Weekly Orders: {weekly_freq.mean():.2f}\n")
        f.write(f"Retention Rate: {retention_rate:.2f}%\n")
        f.write(f"Churn Rate: {churn_rate:.2f}%\n")

    return revenue_per_cust

def load_and_clean_campaign_data(filepath):
    print("\nLoading and Cleaning Campaign Data...")
    df = pd.read_csv(filepath)
    
    # 1. Fix missing/invalid values
    # Remove rows with negative spend or clicks < 0
    df = df[(df['Spend'] >= 0) & (df['Clicks'] >= 0)]
    
    # 2. Validate Clicks (Clicks <= Impressions)
    df = df[df['Clicks'] <= df['Impressions']]
    
    return df

def analyze_campaign_performance(df):
    print("\n--- Part B: Marketing Campaign Performance ---")
    
    # Aggregate by Campaign ID
    camp_agg = df.groupby('Campaign ID').agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Conversions': 'sum',
        'Spend': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    # A. CTR
    camp_agg['CTR'] = (camp_agg['Clicks'] / camp_agg['Impressions']) * 100
    
    # B. Conversion Rate
    camp_agg['ConversionRate'] = (camp_agg['Conversions'] / camp_agg['Clicks']) * 100
    
    # C. Cost Per Lead (CPL) - Assuming Conversion = Lead
    camp_agg['CPL'] = camp_agg['Spend'] / camp_agg['Conversions']
    
    # D. ROI
    camp_agg['ROI'] = ((camp_agg['Revenue'] - camp_agg['Spend']) / camp_agg['Spend']) * 100
    
    # E. CAC (Customer Acquisition Cost)
    # Assuming each conversion is a "New Customer" for this simplified calculation
    # Or we can use the total spend / total conversions for the whole dataset
    total_spend = camp_agg['Spend'].sum()
    total_new_customers = camp_agg['Conversions'].sum()
    overall_cac = total_spend / total_new_customers
    
    print("Campaign Performance Metrics:")
    print(camp_agg[['Campaign ID', 'CTR', 'ConversionRate', 'ROI', 'CPL']])
    print(f"Overall CAC: ${overall_cac:.2f}")
    
    # Visualizations
    
    # 1. CTR Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Campaign ID', y='CTR', data=camp_agg, palette='magma')
    plt.title('CTR by Campaign')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ctr_comparison.png")
    plt.close()
    
    # 2. Spend vs Revenue (Scatter Plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Spend', y='Revenue', data=camp_agg, hue='Campaign ID', s=100)
    # Add diagonal line for break-even
    max_val = max(camp_agg['Spend'].max(), camp_agg['Revenue'].max())
    plt.plot([0, max_val], [0, max_val], '--', color='gray', label='Break-even')
    plt.title('Spend vs Revenue by Campaign')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spend_vs_revenue.png")
    plt.close()
    
    # 3. Conversion Rate Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Campaign ID', y='ConversionRate', data=camp_agg, palette='coolwarm')
    plt.title('Conversion Rate by Campaign')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/conversion_rate.png")
    plt.close()
    
    # 4. ROI Trends (Line Chart - if we had time series, but here we have agg. Let's do Bar for ROI)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Campaign ID', y='ROI', data=camp_agg, palette='RdYlGn')
    plt.title('ROI by Campaign')
    plt.axhline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roi_comparison.png")
    plt.close()
    
    # Calculate overall averages for reporting
    avg_ctr = camp_agg['CTR'].mean()
    avg_conv_rate = camp_agg['ConversionRate'].mean()
    avg_roi = camp_agg['ROI'].mean()
    avg_cpl = camp_agg['CPL'].mean()

    # Append to metrics.txt
    with open('metrics.txt', 'a') as f:
        f.write(f"Overall CAC: ${overall_cac:.2f}\n")
        f.write(f"Average CTR: {avg_ctr:.2f}%\n")
        f.write(f"Average Conversion Rate: {avg_conv_rate:.2f}%\n")
        f.write(f"Average ROI: {avg_roi:.2f}%\n")
        f.write(f"Average CPL: ${avg_cpl:.2f}\n")

    return camp_agg

if __name__ == "__main__":
    # Part A
    if os.path.exists('customer_data.csv'):
        df_cust = load_and_clean_customer_data('customer_data.csv')
        analyze_customer_behavior(df_cust)
    else:
        print("customer_data.csv not found!")
        
    # Part B
    if os.path.exists('campaign_data.csv'):
        df_camp = load_and_clean_campaign_data('campaign_data.csv')
        analyze_campaign_performance(df_camp)
    else:
        print("campaign_data.csv not found!")
