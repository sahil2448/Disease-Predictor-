# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("models/heart_rf_pipeline.pkl")

model = load_model()

st.title("Heart Disease Predictor ❤️")
st.caption("Binary screening model (0 = no disease, 1 = disease). Not a medical diagnosis.")

# --- Single prediction form ---
with st.form("single_input"):
    st.subheader("Single prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 54)
        trestbps = st.number_input("Resting BP (trestbps)", 50, 250, 130)
        chol = st.number_input("Serum Cholesterol (chol)", 50, 700, 246)
        thalach = st.number_input("Max Heart Rate (thalach)", 50, 250, 150)
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain (cp)", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG (restecg)", ["normal", "ST-T abnormality", "left ventricular hypertrophy"])
        exang = st.selectbox("Exercise-induced angina (exang)", ["No", "Yes"])
        slope = st.selectbox("ST segment slope (slope)", ["upsloping", "flat", "downsloping"])
        ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", ["normal", "fixed defect", "reversible defect"])

    submit_single = st.form_submit_button("Predict")

if submit_single:
    row = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,  # adjust if your CSV uses 0/1 codes
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": restecg,
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": int(ca),
        "thal": thal,
    }
    X_user = pd.DataFrame([row])
    pred = model.predict(X_user)[0]
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.named_steps["clf"].predict_proba(X_user)[0, 1]
    else:
        proba = float(pred)
    st.success(f"Prediction: {int(pred)} | Disease probability: {proba:.2f}")

# --- Batch prediction upload ---
st.divider()
st.subheader("Batch prediction (CSV upload)")

def normalize_heart_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # thalch -> thalach
    if "thalch" in df.columns and "thalach" not in df.columns:
        df = df.rename(columns={"thalch": "thalach"})
    # common typos
    if "thal" in df.columns:
        df["thal"] = df["thal"].astype(str).str.strip().replace({"reversable defect": "reversible defect"})
    if "restecg" in df.columns:
        df["restecg"] = df["restecg"].astype(str).str.strip().replace({"lv hypertrophy": "left ventricular hypertrophy"})
    # booleans
    if "exang" in df.columns:
        df["exang"] = df["exang"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(df["exang"])
    # rebuild sex from one-hot if present
    if {"sex_Female", "sex_Male"}.issubset(df.columns) and "sex" not in df.columns:
        df["sex"] = (df["sex_Male"] == 1).astype(int)
        df = df.drop(columns=["sex_Female", "sex_Male"])
    # rebuild cp from one-hot if present
    cp_cols = [c for c in df.columns if c.startswith("cp_")]
    if cp_cols and "cp" not in df.columns:
        def cp_from_onehot(row):
            for c in cp_cols:
                if row.get(c, 0) == 1:
                    return c.replace("cp_", "").strip()
            return "typical angina"
        df["cp"] = df.apply(cp_from_onehot, axis=1)
        df = df.drop(columns=cp_cols)
    required = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    return df[required]

csv_file = st.file_uploader("Upload CSV", type=["csv"])
if csv_file is not None:
    df_raw = pd.read_csv(csv_file)
    df_norm = normalize_heart_csv(df_raw)
    preds = model.predict(df_norm)
    if hasattr(model.named_steps["clf"], "predict_proba"):
        probas = model.named_steps["clf"].predict_proba(df_norm)[:, 1]
    else:
        probas = (preds == 1).astype(float)
    out = df_raw.copy()
    out["Heart_Disease_prediction"] = preds
    out["Disease_probability"] = np.round(probas, 3)
    st.success(f"Predicted {int((preds==1).sum())} positives out of {len(preds)} rows.")
    st.dataframe(out.head(50))
    st.download_button("Download predictions", out.to_csv(index=False), file_name="heart_predictions.csv")
