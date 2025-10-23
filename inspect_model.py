import joblib
import pandas as pd
from app import normalize_heart_csv

model = joblib.load("models/heart_rf_pipeline.pkl")
print("Model pipeline steps:", model.named_steps.keys())

# create a single sample like your app does
sample = {
    "age": 58, "sex": 1, "cp": "typical angina", "trestbps": 130,
    "chol": 220, "fbs": 1, "restecg": "normal", "thalach": 150,
    "exang": 0, "oldpeak": 1.4, "slope": "flat", "ca": 0, "thal": "fixed defect"
}
X_user = pd.DataFrame([sample])
X_user = normalize_heart_csv(X_user)
print("Input columns after normalize:", X_user.columns.tolist())

# try transform to see if preprocessing step raises
pre = model.named_steps["preprocess"]
try:
    feat_array = pre.transform(X_user)  # will throw if shape/dtypes/categories mismatch
    print("Preprocessed array shape:", feat_array.shape)
except Exception as e:
    print("Preprocess transform failed:", e)
    import traceback; traceback.print_exc()

# feature names (if sklearn version supports it)
try:
    names = pre.get_feature_names_out(X_user.columns)
    print("Feature names out (len={}):".format(len(names)))
except Exception as e:
    print("Could not get feature names:", e)
