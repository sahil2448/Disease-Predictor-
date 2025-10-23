# app.py
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tempfile
import traceback


MODEL_PATH = "models/heart_rf_pipeline.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def normalize_heart_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # thalch -> thalach
    if "thalch" in df.columns and "thalach" not in df.columns:
        df = df.rename(columns={"thalch": "thalach"})
    # common typos
    if "thal" in df.columns:
        df["thal"] = (
            df["thal"].astype(str).str.strip().replace({"reversable defect": "reversible defect"})
        )
    if "restecg" in df.columns:
        df["restecg"] = (
            df["restecg"]
            .astype(str)
            .str.strip()
            .replace({"lv hypertrophy": "left ventricular hypertrophy"})
        )
    # booleans
    if "exang" in df.columns:
        ex_map = {"TRUE": 1, "FALSE": 0}
        df["exang"] = df["exang"].astype(str).str.upper().map(ex_map).fillna(df["exang"])
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


def predict_single(age, trestbps, chol, thalach, oldpeak, sex, cp, fbs, restecg, exang, slope, ca, thal):
    try:
        # build raw row similar to your form
        row = {
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": 1 if fbs == "Yes" else 0,
            "restecg": restecg,
            "thalach": thalach,
            "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        # Normalize to the exact schema your pipeline expects
        X_user = pd.DataFrame([row])
        X_user = normalize_heart_csv(X_user)

        # debug prints - will show in your terminal where you launched app.py
        print("DEBUG single input (after normalize):")
        print(X_user.to_dict(orient="records"))

        # predict
        pred = model.predict(X_user)[0]
        if hasattr(model.named_steps["clf"], "predict_proba"):
            proba = float(model.named_steps["clf"].predict_proba(X_user)[0, 1])
        else:
            proba = float(pred)

        return int(pred), round(proba, 3)

    except Exception:
        tb = traceback.format_exc()
        print("ERROR in predict_single:\n", tb)
        # return safe values so Gradio doesn't display the red Error badge.
        return None, None



def predict_batch(file_path: str):
    try:
        df_raw = pd.read_csv(file_path)
        df_norm = normalize_heart_csv(df_raw)

        print("DEBUG batch input columns (after normalize):", df_norm.columns.tolist())
        print("DEBUG batch head:\n", df_norm.head().to_dict(orient="records")[:3])

        preds = model.predict(df_norm)
        if hasattr(model.named_steps["clf"], "predict_proba"):
            probas = model.named_steps["clf"].predict_proba(df_norm)[:, 1]
        else:
            probas = (preds == 1).astype(float)

        out = df_raw.copy()
        out["Heart_Disease_prediction"] = preds
        out["Disease_probability"] = np.round(probas, 3)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        out.to_csv(tmp.name, index=False)
        summary = f"Predicted {int((preds == 1).sum())} positives out of {len(preds)} rows."
        return summary, out, tmp.name

    except Exception:
        tb = traceback.format_exc()
        print("ERROR in predict_batch:\n", tb)
        # For batch, return a short textual error summary + empty values
        return "Error during batch prediction. See server console for details.", pd.DataFrame(), ""
    


    
with gr.Blocks(title="Heart Disease Predictor") as demo:
    gr.Markdown("Binary screening model (0 = no disease, 1 = disease). Not a medical diagnosis.")
    with gr.Tab("Single prediction"):
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Age", value=54, precision=0)
                trestbps = gr.Number(label="Resting BP (trestbps)", value=130, precision=0)
                chol = gr.Number(label="Serum Cholesterol (chol)", value=246, precision=0)
                thalach = gr.Number(label="Max Heart Rate (thalach)", value=150, precision=0)
                oldpeak = gr.Number(label="ST Depression (oldpeak)", value=1.0)
            with gr.Column():
                sex = gr.Dropdown(["Male", "Female"], value="Male", label="Sex")
                cp = gr.Dropdown(
                    ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"],
                    value="typical angina",
                    label="Chest Pain (cp)",
                )
                fbs = gr.Dropdown(["No", "Yes"], value="No", label="Fasting Blood Sugar > 120 mg/dl (fbs)")
                restecg = gr.Dropdown(
                    ["normal", "ST-T abnormality", "left ventricular hypertrophy"],
                    value="normal",
                    label="Resting ECG (restecg)",
                )
                exang = gr.Dropdown(["No", "Yes"], value="No", label="Exercise-induced angina (exang)")
                slope = gr.Dropdown(["upsloping", "flat", "downsloping"], value="flat", label="ST segment slope (slope)")
                ca = gr.Dropdown([0, 1, 2, 3, 4], value=0, label="Major vessels (ca)")
                thal = gr.Dropdown(["normal", "fixed defect", "reversible defect"], value="normal", label="Thalassemia (thal)")
        btn = gr.Button("Predict")
        pred_out = gr.Number(label="Prediction (0/1)")
        proba_out = gr.Number(label="Disease probability")
        btn.click(
            predict_single,
            inputs=[age, trestbps, chol, thalach, oldpeak, sex, cp, fbs, restecg, exang, slope, ca, thal],
            outputs=[pred_out, proba_out],
        )

    with gr.Tab("Batch prediction (CSV)"):
        csv_in = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")
        summary = gr.Markdown()
        df_preview = gr.Dataframe(wrap=True, label="Predictions (preview)")
        file_out = gr.File(label="Download predictions")
        run_batch = gr.Button("Run batch prediction")
        run_batch.click(predict_batch, inputs=[csv_in], outputs=[summary, df_preview, file_out])

if __name__ == "__main__":
    demo.launch()
