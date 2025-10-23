# src/train.py
import pandas as pd
import math
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_PATH = Path("data/heart_dataset.csv")  # your attached file name
# --- robust header normalization + target detection ---
df = pd.read_csv(DATA_PATH)

# normalize column names (strip whitespace and remove BOM/non-printing)
df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

# debug prints (safe to remove later) -- shows exact repr of column names
print("DEBUG: columns (repr):", [repr(c) for c in df.columns])
print("DEBUG: columns list:", df.columns.tolist())

# candidate names (add more variants here if you want)
CANDIDATES = ["num", "target", "output", "HeartDisease", "heart_disease", "Heart_Disease", "label"]

# try exact match first
target_col = next((c for c in CANDIDATES if c in df.columns), None)

# if not found, try a case-insensitive match
if target_col is None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in CANDIDATES:
        if cand.lower() in lower_map:
            target_col = lower_map[cand.lower()]
            break

if target_col is None:
    raise ValueError(
        "No label column found. Add a binary 'target' column (0/1) or the original 'num' 0..4 label to proceed with training. "
        f"Current columns: {list(df.columns)}"
    )

print("Using target column:", repr(target_col))
# --- end robust detection ---

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "heart_rf_pipeline.pkl"

df = pd.read_csv(DATA_PATH)

# --- Rebuild schema from your columns ---
# thalch -> thalach
if "thalch" in df.columns and "thalach" not in df.columns:
    df = df.rename(columns={"thalch": "thalach"})

# Fix typos/variants
if "restecg" in df.columns:
    df["restecg"] = (
        df["restecg"].astype(str).str.strip()
        .replace({"lv hypertrophy": "left ventricular hypertrophy"})
    )
if "thal" in df.columns:
    df["thal"] = (
        df["thal"].astype(str).str.strip()
        .replace({"reversable defect": "reversible defect"})
    )
if "exang" in df.columns:
    df["exang"] = (
        df["exang"].astype(str).str.upper()
        .map({"TRUE": 1, "FALSE": 0})
        .fillna(df["exang"])
    )

# Rebuild sex from one-hot if present
if {"sex_Female", "sex_Male"}.issubset(df.columns) and "sex" not in df.columns:
    df["sex"] = (df["sex_Male"] == 1).astype(int)
    df = df.drop(columns=["sex_Female", "sex_Male"])

# Rebuild cp from one-hot if present
cp_cols = [c for c in df.columns if c.startswith("cp_")]
if cp_cols and "cp" not in df.columns:
    # map your one-hot names to the app's expected labels
    cp_label_map = {
        "cp_typical angina": "typical angina",
        "cp_atypical angina": "atypical angina",
        "cp_non-anginal": "non-anginal pain",
        "cp_asymptomatic": "asymptomatic",
    }
    def cp_from_onehot(row):
        for c in cp_cols:
            if row.get(c, 0) == 1:
                return cp_label_map.get(c, c.replace("cp_", "").strip())
        return "typical angina"
    df["cp"] = df.apply(cp_from_onehot, axis=1)
    df = df.drop(columns=cp_cols)

# --- Detect label ---
CANDIDATES = ["num", "target", "output", "HeartDisease", "heart_disease", "Heart_Disease", "label"]
target_col = next((c for c in CANDIDATES if c in df.columns), None)

if target_col is None:
    raise ValueError(
        "No label column found. Add a binary 'target' column (0/1) or the original 'num' 0..4 label to proceed with training. "
        f"Current columns: {list(df.columns)}"
    )

y_raw = df[target_col]
# If it's UCI style 0..4, binarize; else assume already binary
try:
    uniq = set(pd.unique(pd.to_numeric(y_raw, errors="coerce").dropna()))
except Exception:
    uniq = set(pd.unique(y_raw.dropna()))
if uniq.issubset({0,1,2,3,4}):
    y = (pd.to_numeric(y_raw, errors="coerce") > 0).astype(int)
else:
    y = pd.to_numeric(y_raw, errors="coerce").astype(int)

X = df.drop(columns=[target_col])

# Split numeric vs categorical as produced above
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
])


# desired test_size (can be float or int)
requested_test_size = 0.2

n_samples = len(y)
n_classes = y.nunique(dropna=True)

# if requested_test_size is a float, ensure at least one sample per class in test set
if isinstance(requested_test_size, float):
    n_test = math.ceil(requested_test_size * n_samples)
    if n_test < n_classes:
        requested_test_size = n_classes / n_samples
        print(f"Adjusted test_size to {requested_test_size:.3f} to have at least one sample per class in the test set.")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=requested_test_size, random_state=42, stratify=y
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
