# train.py
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
import joblib

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "sample_data.csv"
MODEL_PATH = MODELS_DIR / "gbm_pipeline.pkl"

def generate_sample_data(path, n=5000, random_state=42):
    """Generate a synthetic tabular binary classification dataset and save to CSV."""
    rng = np.random.RandomState(random_state)
    # numeric features
    age = rng.randint(18, 80, size=n)
    income = np.round(rng.normal(50000, 20000, size=n)).astype(int)
    balance = np.round(rng.normal(2000, 1500, size=n)).astype(int)
    # categorical
    city = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    has_credit_card = rng.choice([0,1], size=n, p=[0.4, 0.6])
    # target (simple synthetic rule + randomness)
    score = 0.02*(age) + 0.00002*income - 0.0004*balance + (has_credit_card * 0.5) + (city == "A")*0.2
    prob = 1 / (1 + np.exp(- (score - 0.5) ))
    target = (rng.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "balance": balance,
        "city": city,
        "has_credit_card": has_credit_card,
        "target": target
    })
    df.to_csv(path, index=False)
    print(f"Generated sample data -> {path}")

def build_and_train(csv_path, model_path):
    df = pd.read_csv(csv_path)
    # simple train/test split
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Column lists
    numeric_features = ["age", "income", "balance"]
    categorical_features = ["city", "has_credit_card"]  # has_credit_card treated as categorical here

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("gbm", gbm)
    ])

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # evaluate
    val_preds = pipeline.predict(X_val)
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    print("Validation classification report:")
    print(classification_report(y_val, val_preds))
    try:
        print("Validation ROC AUC:", roc_auc_score(y_val, val_proba))
    except ValueError as e:
        logging.warning(f"Could not calculate ROC AUC: {e}")

    # save pipeline
    joblib.dump(pipeline, model_path)
    print(f"Saved model pipeline -> {model_path}")

if __name__ == "__main__":
    if not CSV_PATH.exists():
        generate_sample_data(CSV_PATH)
    build_and_train(CSV_PATH, MODEL_PATH)
