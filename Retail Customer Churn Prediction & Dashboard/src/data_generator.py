# src/data_generator.py

import pandas as pd
import numpy as np

def generate_data(file_path, num_customers=1000, seed=42):
    np.random.seed(seed)
    
    # Generate simulated customer data
    data = pd.DataFrame({
        'customer_id': range(1, num_customers + 1),
        'purchase_frequency': np.random.poisson(lam=5, size=num_customers),
        'recency': np.random.randint(1, 365, size=num_customers),  # Days since last purchase
        'total_spend': np.random.uniform(100, 10000, size=num_customers),
        'complaint_history': np.random.randint(0, 5, size=num_customers),
        'signup_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, num_customers), unit='D')
    })
    
    # Simulate churn: customers with low purchase frequency, high recency, or high complaint history are more likely to churn
    data['churn'] = ((data['purchase_frequency'] < 3) & (data['recency'] > 200)) | (data['complaint_history'] >= 3)
    data['churn'] = data['churn'].astype(int)
    
    # Save the generated dataset as a CSV file
    data.to_csv(file_path, index=False)
    print(f"Data generated and saved to {file_path}")
