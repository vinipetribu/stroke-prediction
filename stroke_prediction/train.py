import joblib
import json # <-- Nova importação necessária
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" # <-- Nova pasta para salvar os relatórios

def train_model():
    print("[1/4] Carregando features e dividindo dados...")
    df = pd.read_csv(FEATURES_PATH)
    
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("[2/4] Construindo o Super Pipeline...")
    pipeline = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    print("[3/4] Treinando o pipeline...")
    pipeline.fit(X_train, y_train)
    
    print("\n--- Relatório de Classificação (Dados de Teste) ---")
    y_pred = pipeline.predict(X_test)
    
    # Imprime no terminal para você ler na hora
    print(classification_report(y_test, y_pred))
    
    # Gera a versão em dicionário para salvar no JSON
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    print("\n[4/4] Salvando os artefatos...")
    
    # 1. Salva o Modelo .pkl
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "logistic_regression.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Modelo salvo em: {model_path}")

    # 2. Salva as Métricas .json
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "metrics.json"
    
    with open(metrics_path, "w") as f:
        # O indent=4 deixa o JSON bonito e fácil de ler (formatado)
        json.dump(report_dict, f, indent=4) 
        
    print(f"Métricas salvas em: {metrics_path}")

if __name__ == "__main__":
    train_model()