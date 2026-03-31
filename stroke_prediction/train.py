import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def build_pipeline(name: str) -> Pipeline:
    if name == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                (
                    "classifier",
                    LogisticRegression(random_state=42, max_iter=1000),
                ),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=42, n_estimators=200, n_jobs=-1
                    ),
                ),
            ]
        )
    if name == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        random_state=42,
                        max_iter=200,
                        learning_rate=0.05,
                    ),
                ),
            ]
        )
    raise ValueError(f"Modelo desconhecido: {name}")


def train_models():
    print("[1/3] Carregando features e dividindo dados...")
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: dict = {}

    specs = [
        ("logistic_regression", "Regressão logística"),
        ("random_forest", "Random Forest"),
        (
            "hist_gradient_boosting",
            "HistGradientBoosting (gradient boosting, sklearn)",
        ),
    ]

    for key, label in specs:
        print(f"\n[2/3] Treinando: {label}...")
        pipeline = build_pipeline(key)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"\n--- {label} | classificação (teste) ---")
        print(classification_report(y_test, y_pred))
        all_metrics[key] = classification_report(y_test, y_pred, output_dict=True)

        path = MODEL_DIR / f"{key}.pkl"
        joblib.dump(pipeline, path)
        print(f"Modelo salvo em: {path}")

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n[3/3] Métricas (todos os modelos) em: {metrics_path}")


if __name__ == "__main__":
    train_models()
