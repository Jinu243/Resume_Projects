# src/feature_engineering.py

import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Engineer additional features for churn prediction.
    """
    df = df.copy()
    
    # Convert signup_date to a month-year format for monthly trend analysis
    df['signup_month'] = df['signup_date'].dt.to_period('M').astype(str)
    
    # Add a feature to reduce skewness in total_spend
    df['log_total_spend'] = np.log1p(df['total_spend'])
    
    # Additional feature engineering steps can be added here
    
    return df
