# run.py

import os
from src import data_generator, data_processing, feature_engineering, model, dashboard

DATA_PATH = os.path.join("data", "simulated_data.csv")
MODEL_PATH = os.path.join("data", "churn_model.pkl")

def main():
    # Step 1: Generate Data if not already available
    if not os.path.exists(DATA_PATH):
        print("Generating simulated data...")
        data_generator.generate_data(DATA_PATH, num_customers=1000)
    else:
        print("Data file already exists.")
        
    # Step 2: Load and process data
    print("Processing data...")
    df = data_processing.load_data(DATA_PATH)
    
    # Step 3: Feature engineering
    print("Engineering features...")
    df_features = feature_engineering.engineer_features(df)
    
    # Step 4: Train the churn prediction model
    print("Training model...")
    clf, X_test, y_test, y_pred = model.train_model(df_features)
    
    # Save the trained model
    model.save_model(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Output model evaluation metrics
    print("\nModel Evaluation:")
    model.evaluate_model(y_test, y_pred)
    
    # Step 5: Generate dashboard visualizations
    print("Generating dashboard visualizations...")
    dashboard.generate_dashboard(df_features, clf)

if __name__ == '__main__':
    main()
