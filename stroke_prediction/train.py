import json
import os
from pathlib import Path

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import mlflow
except ImportError:
    mlflow = None

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
                    RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1),
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


def _log_classification_metrics(report: dict, prefix: str = "test") -> None:
    if mlflow is None:
        return
    if "accuracy" in report:
        mlflow.log_metric(f"{prefix}_accuracy", float(report["accuracy"]))
    for label_key in ("0", "1"):
        block = report.get(label_key)
        if isinstance(block, dict):
            mlflow.log_metric(f"{prefix}_precision_{label_key}", float(block["precision"]))
            mlflow.log_metric(f"{prefix}_recall_{label_key}", float(block["recall"]))
            mlflow.log_metric(f"{prefix}_f1_{label_key}", float(block["f1-score"]))
    for avg_name in ("macro avg", "weighted avg"):
        block = report.get(avg_name)
        if isinstance(block, dict):
            slug = avg_name.replace(" ", "_")
            for m_key, m_name in (
                ("precision", "precision"),
                ("recall", "recall"),
                ("f1-score", "f1"),
            ):
                if m_key in block:
                    mlflow.log_metric(f"{prefix}_{slug}_{m_name}", float(block[m_key]))


def _setup_mlflow() -> bool:
    if mlflow is None:
        print("[MLflow] Pacote não instalado (pip install mlflow).")
        return False
    if os.environ.get("MLFLOW_DISABLE", "").lower() in ("1", "true", "yes"):
        print("[MLflow] Desativado (MLFLOW_DISABLE).")
        return False
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5002")
    exp = os.environ.get("MLFLOW_EXPERIMENT_NAME", "stroke-prediction")
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(exp)
        print(f"[MLflow] Tracking: {uri} | experimento: {exp}")
        return True
    except Exception as e:
        print(f"[MLflow] Não foi possível ligar a {uri} ({e}). Treino continua só em disco.")
        return False


def _select_champion(all_metrics: dict, model_uris: dict[str, str] | None = None) -> dict:
    champion_name, champion_report = max(
        all_metrics.items(),
        key=lambda item: item[1].get("1", {}).get("recall", 0.0),
    )
    positive_class_metrics = champion_report.get("1", {})
    champion = {
        "model_name": champion_name,
        "model_path": f"models/{champion_name}.pkl",
        "selection_metric": "recall for positive class (stroke=1)",
        "selection_metric_value": positive_class_metrics.get("recall", 0.0),
        "reason": (
            "Chosen as champion because it has the best recall for the minority positive "
            "class among the trained models, which is more appropriate for stroke screening "
            "than raw accuracy on this imbalanced dataset."
        ),
        "compared_models": {
            model_name: {
                "recall_stroke_1": report.get("1", {}).get("recall", 0.0),
                "precision_stroke_1": report.get("1", {}).get("precision", 0.0),
                "accuracy": report.get("accuracy", 0.0),
            }
            for model_name, report in all_metrics.items()
        },
    }
    if model_uris and champion_name in model_uris:
        champion["mlflow_model_uri"] = model_uris[champion_name]
    return champion


def _register_champion_model(champion: dict) -> dict:
    if mlflow is None or "mlflow_model_uri" not in champion:
        return champion

    registered_model_name = os.environ.get(
        "MLFLOW_REGISTERED_MODEL_NAME", "stroke-prediction-champion"
    )
    try:
        result = mlflow.register_model(champion["mlflow_model_uri"], registered_model_name)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(registered_model_name, "champion", result.version)

        champion["mlflow_registered_model_name"] = registered_model_name
        champion["mlflow_registered_model_version"] = result.version
        champion["mlflow_model_uri"] = f"models:/{registered_model_name}@champion"
        print(f"Champion registrado no MLflow Model Registry: {champion['mlflow_model_uri']}")
    except Exception as e:
        print(
            "[MLflow] Nao foi possivel registrar o champion no Model Registry "
            f"({e}). Usando URI do run."
        )

    return champion


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
    model_uris: dict[str, str] = {}
    use_mlflow = _setup_mlflow()

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

        if use_mlflow:
            with mlflow.start_run(run_name=key):
                mlflow.log_param("model_key", key)
                mlflow.log_param("model_label", label)
                mlflow.log_param("random_state_split", 42)
                mlflow.log_param("test_size", 0.2)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                print(f"\n--- {label} | classificação (teste) ---")
                print(classification_report(y_test, y_pred))
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                all_metrics[key] = report_dict
                _log_classification_metrics(report_dict)
                mlflow.sklearn.log_model(pipeline, artifact_path="model")
                model_uris[key] = f"runs:/{mlflow.active_run().info.run_id}/model"
        else:
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

    champion = _register_champion_model(_select_champion(all_metrics, model_uris))
    champion_path = REPORTS_DIR / "champion.json"
    with open(champion_path, "w") as f:
        json.dump(champion, f, indent=4)
    print(f"Champion selecionado: {champion['model_name']} ({champion_path})")


if __name__ == "__main__":
    train_models()
