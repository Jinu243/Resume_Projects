# src/dashboard.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_dashboard(df, model):
    """
    Generate and save visualizations for the dashboard.
    """
    # --- Feature Importance Plot ---
    features = ['purchase_frequency', 'recency', 'total_spend', 'complaint_history', 'log_total_spend']
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=features).sort_values(ascending=False)
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=feat_importances.index, y=feat_importances.values)
    plt.title("Feature Importance in Churn Prediction")
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.savefig("dashboard_feature_importance.png")
    plt.show()
    
    # --- Monthly Churn Trends ---
    monthly_churn = df.groupby('signup_month')['churn'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_churn, x='signup_month', y='churn', marker="o")
    plt.title("Monthly Churn Trends")
    plt.ylabel("Average Churn Rate")
    plt.xlabel("Signup Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("dashboard_monthly_churn.png")
    plt.show()
    
    # --- High-risk Customer Analysis ---
    # Plot distribution of purchase frequency among high-risk customers (churn = 1)
    high_risk = df[df['churn'] == 1]
    plt.figure(figsize=(10, 5))
    sns.histplot(high_risk['purchase_frequency'], bins=20, kde=True)
    plt.title("High-risk Customers - Purchase Frequency Distribution")
    plt.xlabel("Purchase Frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("dashboard_high_risk_purchase_frequency.png")
    plt.show()
    
    print("Dashboard visualizations generated and saved as PNG files.")
