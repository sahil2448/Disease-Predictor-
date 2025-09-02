from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and scaler
model = None
scaler = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    try:
        # In a real deployment, these would be proper file paths
        # For now, we'll create a mock model
        print("Loading model and scaler...")
        # model = joblib.load('heart_rf_model.pkl')
        # scaler = joblib.load('hear_scalar.pkl')
        
        # Mock model for demonstration
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        
        # Create dummy training data to fit the scaler
        dummy_data = np.random.randn(100, 29)  # 29 features after one-hot encoding
        scaler.fit(dummy_data)
        
        print("Model and scaler loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_input(data):
    """Preprocess input data to match training format"""
    try:
        # Create a DataFrame with the expected columns
        feature_columns = [
            'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca',
            'sex_Female', 'sex_Male', 'cp_asymptomatic', 'cp_atypical angina',
            'cp_non-anginal', 'cp_typical angina', 'fbs_False', 'fbs_True',
            'restecg_lv hypertrophy', 'restecg_normal', 'restecg_st-t abnormality',
            'exang_False', 'exang_True', 'slope_downsloping', 'slope_flat',
            'slope_upsloping', 'thal_fixed defect', 'thal_normal',
            'thal_reversable defect'
        ]
        
        # Initialize all features to 0
        processed_data = {col: 0 for col in feature_columns}
        
        # Fill in the numeric features
        processed_data['age'] = data.get('age', 0)
        processed_data['trestbps'] = data.get('trestbps', 0)
        processed_data['chol'] = data.get('chol', 0)
        processed_data['thalch'] = data.get('thalch', 0)
        processed_data['oldpeak'] = data.get('oldpeak', 0)
        processed_data['ca'] = data.get('ca', 0)
        
        # One-hot encode categorical features
        sex = data.get('sex', '')
        if sex == 'Female':
            processed_data['sex_Female'] = 1
        elif sex == 'Male':
            processed_data['sex_Male'] = 1
            
        cp = data.get('cp', '')
        if cp == 'asymptomatic':
            processed_data['cp_asymptomatic'] = 1
        elif cp == 'atypical angina':
            processed_data['cp_atypical angina'] = 1
        elif cp == 'non-anginal':
            processed_data['cp_non-anginal'] = 1
        elif cp == 'typical angina':
            processed_data['cp_typical angina'] = 1
            
        fbs = data.get('fbs', False)
        if fbs:
            processed_data['fbs_True'] = 1
        else:
            processed_data['fbs_False'] = 1
            
        restecg = data.get('restecg', '')
        if restecg == 'lv hypertrophy':
            processed_data['restecg_lv hypertrophy'] = 1
        elif restecg == 'normal':
            processed_data['restecg_normal'] = 1
        elif restecg == 'st-t abnormality':
            processed_data['restecg_st-t abnormality'] = 1
            
        exang = data.get('exang', False)
        if exang:
            processed_data['exang_True'] = 1
        else:
            processed_data['exang_False'] = 1
            
        slope = data.get('slope', '')
        if slope == 'downsloping':
            processed_data['slope_downsloping'] = 1
        elif slope == 'flat':
            processed_data['slope_flat'] = 1
        elif slope == 'upsloping':
            processed_data['slope_upsloping'] = 1
            
        thal = data.get('thal', '')
        if thal == 'fixed defect':
            processed_data['thal_fixed defect'] = 1
        elif thal == 'normal':
            processed_data['thal_normal'] = 1
        elif thal == 'reversable defect':
            processed_data['thal_reversable defect'] = 1
        
        # Convert to DataFrame
        df = pd.DataFrame([processed_data])
        return df
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict heart disease risk"""
    try:
        # Get input data
        input_data = request.json
        
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Preprocess the input
        processed_df = preprocess_input(input_data)
        
        # Scale the data
        if scaler is None:
            return jsonify({"error": "Scaler not loaded"}), 500
            
        scaled_data = scaler.transform(processed_df)
        
        # Make prediction (mock for now)
        # prediction = model.predict(scaled_data)[0]
        # probability = model.predict_proba(scaled_data)[0][1]
        
        # Mock prediction based on risk factors
        risk_score = calculate_risk_score(input_data)
        prediction = 1 if risk_score > 0.5 else 0
        probability = risk_score
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        # Calculate confidence (mock)
        confidence = 0.85 + np.random.random() * 0.1
        
        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "confidence": float(confidence)
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

def calculate_risk_score(data):
    """Calculate risk score based on input features"""
    score = 0.0
    
    # Age factor
    age = data.get('age', 0)
    if age > 60:
        score += 0.3
    elif age > 45:
        score += 0.2
    else:
        score += 0.1
    
    # Blood pressure
    trestbps = data.get('trestbps', 0)
    if trestbps > 140:
        score += 0.25
    elif trestbps > 120:
        score += 0.15
    
    # Cholesterol
    chol = data.get('chol', 0)
    if chol > 240:
        score += 0.2
    elif chol > 200:
        score += 0.1
    
    # Chest pain
    cp = data.get('cp', '')
    if cp == 'typical angina':
        score += 0.3
    elif cp == 'atypical angina':
        score += 0.2
    
    # Exercise angina
    if data.get('exang', False):
        score += 0.2
    
    # ST depression
    oldpeak = data.get('oldpeak', 0)
    if oldpeak > 2:
        score += 0.2
    elif oldpeak > 1:
        score += 0.1
    
    # Major vessels
    ca = data.get('ca', 0)
    if ca > 0:
        score += ca * 0.15
    
    # Thalassemia
    thal = data.get('thal', '')
    if thal in ['fixed defect', 'reversable defect']:
        score += 0.2
    
    return min(max(score, 0), 1)

if __name__ == '__main__':
    print("Starting Heart Disease Predictor API...")
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)