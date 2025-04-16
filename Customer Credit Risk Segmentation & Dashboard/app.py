import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to load and preprocess the dataset
def load_data():
    # Load dataset from CSV file
    data = pd.read_csv("data/bank_loan_data.csv")
    
    # Feature Engineering: Calculate Loan-to-Income Ratio
    data["Loan_to_Income_Ratio"] = data["LoanAmount"] / data["Income"]
    
    return data

# Function to perform customer segmentation using KMeans clustering
def segment_customers(df, n_clusters=3):
    # Select features for clustering
    features = df[["Age", "Income", "CreditScore", "Loan_to_Income_Ratio", "PreviousDefaults"]]
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(features_scaled)
    
    return df, kmeans

# Function to assign risk levels based on average credit score per cluster
def assign_risk_level(df):
    # Calculate average credit score per cluster
    cluster_scores = df.groupby("Cluster")["CreditScore"].mean()
    
    # Sort clusters: highest average credit score is considered "Low Risk"
    sorted_clusters = cluster_scores.sort_values(ascending=False).index.tolist()
    risk_labels = ["Low Risk", "Medium Risk", "High Risk"]
    
    # Map clusters to risk levels; if there are more clusters, extra clusters are assigned "Medium Risk"
    risk_mapping = {}
    for i, cluster in enumerate(sorted_clusters):
        risk_mapping[cluster] = risk_labels[i] if i < len(risk_labels) else "Medium Risk"
    
    df["Risk_Level"] = df["Cluster"].map(risk_mapping)
    return df

# Main function to run the Streamlit dashboard
def main():
    st.title("Customer Credit Risk Segmentation Dashboard")
    
    # Load the data
    data = load_data()
    st.write("### Original Data")
    st.dataframe(data.head())
    
    # Sidebar control for number of clusters
    st.sidebar.header("Segmentation Settings")
    n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=5, value=3, step=1)
    
    # Create a copy for segmentation
    segmented_data, kmeans_model = segment_customers(data.copy(), n_clusters=n_clusters)
    
    # Assign risk levels based on clustering results
    segmented_data = assign_risk_level(segmented_data)
    
    st.write("### Segmented Data with Risk Levels")
    st.dataframe(segmented_data.head(10))
    
    # Plot: Distribution of clusters
    st.write("### Cluster Distribution")
    cluster_count = segmented_data["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(cluster_count.index.astype(str), cluster_count.values, color="skyblue")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)
    
    # Plot: Average Credit Score per Cluster
    st.write("### Average Credit Score by Cluster")
    avg_score = segmented_data.groupby("Cluster")["CreditScore"].mean()
    fig2, ax2 = plt.subplots()
    ax2.bar(avg_score.index.astype(str), avg_score.values, color="coral")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Average Credit Score")
    st.pyplot(fig2)
    
    # Plot: Risk Level Distribution
    st.write("### Risk Level Distribution")
    risk_count = segmented_data["Risk_Level"].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.bar(risk_count.index, risk_count.values, color="seagreen")
    ax3.set_xlabel("Risk Level")
    ax3.set_ylabel("Number of Customers")
    st.pyplot(fig3)
    
    st.write("#### Interpretation Notes:")
    st.markdown("""
    - **Low Risk:** Customers in this segment have higher average credit scores.
    - **Medium Risk:** Customers in this segment have moderate credit scores.
    - **High Risk:** Customers in this segment have lower average credit scores and possibly other concerning features.
    """)

if __name__ == "__main__":
    main()
