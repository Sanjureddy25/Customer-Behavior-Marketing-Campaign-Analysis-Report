import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_customer_data(num_customers=1000, num_orders=5000):
    print("Generating Customer Data...")
    
    # 1. Generate Customers
    customer_ids = [f'CUST_{i:04d}' for i in range(1, num_customers + 1)]
    signup_start = datetime(2023, 1, 1)
    signup_end = datetime(2024, 6, 30)
    
    customers = []
    for cid in customer_ids:
        signup_date = signup_start + timedelta(days=random.randint(0, (signup_end - signup_start).days))
        # Last active date is sometime after signup, up to today (let's say 2024-12-01)
        max_date = datetime(2024, 12, 1)
        if signup_date > max_date:
            signup_date = max_date - timedelta(days=1)
            
        days_since_signup = (max_date - signup_date).days
        last_active = signup_date + timedelta(days=random.randint(0, days_since_signup))
        
        customers.append({
            'Customer ID': cid,
            'Signup Date': signup_date,
            'Last Active Date': last_active
        })
    
    df_customers = pd.DataFrame(customers)
    
    # 2. Generate Orders
    products = ['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']
    orders = []
    
    for _ in range(num_orders):
        customer = random.choice(customers)
        cid = customer['Customer ID']
        signup_date = customer['Signup Date']
        last_active = customer['Last Active Date']
        
        # Order date must be between signup and last active
        if last_active > signup_date:
            order_date = signup_date + timedelta(days=random.randint(0, (last_active - signup_date).days))
        else:
            order_date = signup_date
            
        amount = round(random.uniform(10, 500), 2)
        # Introduce some negative values to be cleaned later
        if random.random() < 0.02:
            amount = amount * -1
            
        product = random.choice(products)
        
        orders.append({
            'Customer ID': cid,
            'Order Date': order_date,
            'Order Amount': amount,
            'Product Category': product
        })
        
    df_orders = pd.DataFrame(orders)
    
    # Merge to create the final dataset
    # We want a flat structure as per requirements: Customer ID, Order date, Order amount, Product, Activity dates, Signup date
    df_final = pd.merge(df_orders, df_customers, on='Customer ID', how='left')
    
    # Add some duplicates to be cleaned
    df_final = pd.concat([df_final, df_final.sample(n=50)], ignore_index=True)
    
    # Save
    df_final.to_csv('customer_data.csv', index=False)
    print(f"Saved customer_data.csv with {len(df_final)} rows.")
    return df_final

def generate_campaign_data(num_campaigns=10):
    print("Generating Campaign Data...")
    
    campaigns = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(1, num_campaigns + 1):
        camp_id = f'CAMP_{i:02d}'
        # Each campaign runs for a month or so
        camp_start = start_date + timedelta(days=(i-1)*30)
        
        # Generate daily data for the campaign
        duration = 30
        for day in range(duration):
            date = camp_start + timedelta(days=day)
            
            impressions = random.randint(1000, 10000)
            clicks = int(impressions * random.uniform(0.01, 0.05)) # 1% to 5% CTR
            
            # Introduce some missing/invalid clicks
            if random.random() < 0.05:
                clicks = -1 # Invalid
            
            conversions = int(clicks * random.uniform(0.05, 0.20)) if clicks > 0 else 0
            spend = round(clicks * random.uniform(0.5, 2.0), 2) if clicks > 0 else 0 # CPC between 0.5 and 2
            
            # Revenue is usually higher than spend for good campaigns, lower for bad ones
            roi_factor = random.uniform(0.5, 3.0)
            revenue = round(spend * roi_factor, 2)
            
            campaigns.append({
                'Campaign ID': camp_id,
                'Date': date,
                'Impressions': impressions,
                'Clicks': clicks,
                'Conversions': conversions,
                'Spend': spend,
                'Revenue': revenue
            })
            
    df_campaigns = pd.DataFrame(campaigns)
    
    # Save
    df_campaigns.to_csv('campaign_data.csv', index=False)
    print(f"Saved campaign_data.csv with {len(df_campaigns)} rows.")
    return df_campaigns

if __name__ == "__main__":
    generate_customer_data()
    generate_campaign_data()
