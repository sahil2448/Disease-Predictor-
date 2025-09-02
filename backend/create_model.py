"""
Script to recreate the trained model and scaler from the notebook
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def create_model():
    """Recreate the model from the heart disease dataset"""
    
    # Load the dataset
    df = pd.read_csv('heart_dataset - heart_dataset.csv.csv')
    
    # Handle missing values for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Get categorical columns (excluding target)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Prepare features and target
    # Since the uploaded dataset doesn't have 'num' column, we'll create a synthetic target
    # based on multiple risk factors for demonstration
    X = df.copy()
    
    # Create a synthetic target based on risk factors
    # This is a simplified approach for demonstration
    risk_score = 0
    
    # Age risk (higher age = higher risk)
    if 'age' in X.columns:
        risk_score += (X['age'] > 55).astype(int) * 2
    
    # Cholesterol risk
    if 'chol' in X.columns:
        risk_score += (X['chol'] > 240).astype(int) * 2
    
    # Blood pressure risk
    if 'trestbps' in X.columns:
        risk_score += (X['trestbps'] > 140).astype(int) * 2
    
    # Heart rate risk (lower max heart rate = higher risk)
    if 'thalch' in X.columns:
        risk_score += (X['thalch'] < 120).astype(int) * 2
    
    # Exercise angina risk
    if 'exang' in X.columns:
        risk_score += (X['exang'] == True).astype(int) * 3
    
    # ST depression risk
    if 'oldpeak' in X.columns:
        risk_score += (X['oldpeak'] > 1.0).astype(int) * 2
    
    # Create binary target (1 if risk_score >= 4, 0 otherwise)
    y = (risk_score >= 4).astype(int)
    
    print(f"Target distribution: {y.value_counts()}")
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=cat_cols)
    
    print(f"Feature columns after encoding: {X_encoded.columns.tolist()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and scaler
    joblib.dump(rf_model, 'heart_rf_model.pkl')
    joblib.dump(scaler, 'hear_scalar.pkl')
    
    # Create and save template
    template = X_encoded.head(1)
    template.to_csv('Heart_user_template.csv', index=False)
    
    print("\nModel, scaler, and template saved successfully!")
    print(f"Template shape: {template.shape}")
    print(f"Template columns: {template.columns.tolist()}")
    
    return rf_model, scaler, X_encoded.columns.tolist()

if __name__ == "__main__":
    create_model()