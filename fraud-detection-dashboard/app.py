import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    # Load dataset
    data = pd.read_csv("data/transactions.csv")
    
    # Convert Timestamp to datetime
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    
    # Optional: Extract hour of day as a feature (could influence fraud pattern)
    data["Hour"] = data["Timestamp"].dt.hour
    
    return data

def preprocess_data(df):
    # Select features to use: Amount and Hour plus categorical variables
    # Encode categorical features: Location, MerchantID, and Device using one-hot encoding
    categorical_cols = ["Location", "MerchantID", "Device"]
    df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    
    # Use numeric features: Amount and Hour
    df_numeric = df[["Amount", "Hour"]].reset_index(drop=True)
    
    # Combine numeric and encoded categorical features
    features = pd.concat([df_numeric, df_encoded.reset_index(drop=True)], axis=1)
    return features

# -------------------------------
# Anomaly Detection Model
# -------------------------------
def detect_anomalies(features, contamination):
    # Set up Isolation Forest with user-defined contamination rate
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(features)
    
    # Predict anomalies: returns -1 for anomaly, 1 for inlier
    predictions = model.predict(features)
    
    # Also compute anomaly scores (the lower, the more anomalous)
    scores = model.decision_function(features)
    
    return predictions, scores, model

# -------------------------------
# Main Streamlit App
# -------------------------------
def main():
    st.title("Fraud Detection & Transaction Anomaly Analysis")
    
    st.markdown("""
    This dashboard applies an **Isolation Forest** model to identify anomalous transactions that might indicate fraudulent activity.
    Adjust the **contamination** rate (estimated fraction of anomalies) from the sidebar to see different detection sensitivities.
    """)
    
    # Load and display the raw data
    data = load_data()
    st.write("### Raw Transaction Data", data.head())
    
    # Preprocess data for anomaly detection
    features = preprocess_data(data)
    
    # Sidebar: Set contamination rate
    st.sidebar.header("Anomaly Detection Settings")
    contamination = st.sidebar.slider("Contamination (Estimated Fraction of Anomalies)", 
                                      min_value=0.01, max_value=0.3, value=0.1, step=0.01)
    
    # Run anomaly detection
    predictions, scores, model = detect_anomalies(features, contamination)
    
    # Add the predictions and anomaly scores back to the original data
    data["Anomaly_Flag"] = np.where(predictions == -1, "Anomalous", "Normal")
    data["Anomaly_Score"] = scores
    
    st.write("### Transaction Data with Anomaly Flags")
    st.dataframe(data.sort_values("Anomaly_Flag", ascending=False).reset_index(drop=True))
    
    # -------------------------------
    # Visualizations
    # -------------------------------
    st.write("### Distribution of Anomaly Scores")
    fig1, ax1 = plt.subplots()
    ax1.hist(scores, bins=20, color="lightcoral", edgecolor="black")
    ax1.set_xlabel("Anomaly Score")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)
    
    st.write("### Count of Normal vs Anomalous Transactions")
    flag_counts = data["Anomaly_Flag"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.bar(flag_counts.index, flag_counts.values, color=["mediumseagreen", "salmon"])
    ax2.set_xlabel("Transaction Status")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)
    
    st.write("#### Interpretation Notes:")
    st.markdown("""
    - **Anomalous Transactions:** Marked as "Anomalous" by the model, these records have lower decision function scores.
    - **Normal Transactions:** Records that fall within the typical range based on the selected features.
    - Adjust the contamination slider to understand how the model's sensitivity changes with different assumptions of the anomalous rate.
    """)
    
if __name__ == "__main__":
    main()
