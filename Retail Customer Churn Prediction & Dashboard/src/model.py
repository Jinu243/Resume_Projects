# src/model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(df):
    """
    Train a Random Forest model using engineered features.
    Returns the trained model, test data, true labels, and predictions.
    """
    features = ['purchase_frequency', 'recency', 'total_spend', 'complaint_history', 'log_total_spend']
    X = df[features]
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    return clf, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred):
    """
    Evaluate and print the performance of the model.
    """
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy: {:.2f}%".format(accuracy * 100))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
def save_model(model, file_path):
    """
    Save the trained model to disk.
    """
    joblib.dump(model, file_path)
