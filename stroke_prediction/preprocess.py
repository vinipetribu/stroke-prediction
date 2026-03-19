"""
preprocess.py
=============
Pipeline de pré-processamento para o Stroke Prediction Dataset.
Saída: X_train, X_test, y_train, y_test prontos para modelagem.

Estrutura de pastas esperada (Cookiecutter Data Science):
    data/
    ├── raw/          ← dataset original (imutável)
    ├── interim/      ← dados parcialmente transformados
    └── processed/    ← dados finais prontos para modelagem

Uso direto (script):
    python stroke_prediction/preprocess.py

Uso como módulo:
    from stroke_prediction.preprocess import load_and_preprocess
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess()
    
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# PATHS — relativos à raiz do projeto
# ─────────────────────────────────────────────

# Raiz do projeto: dois níveis acima deste arquivo
#   stroke_prediction/preprocess.py → sobe 2x → raiz/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"


# ─────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────

def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carrega o CSV e faz correções iniciais de tipo.

    Parâmetros
    ----------
    path : Path
        Caminho para o CSV. Padrão: data/raw/healthcare-dataset-stroke-data.csv
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em: {path}\n"
            f"Coloque o arquivo em: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(path)

    # bmi pode vir como string 'N/A' dependendo da versão do arquivo
    if df["bmi"].dtype == object:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    return df


# ─────────────────────────────────────────────
# 2. LIMPEZA
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas desnecessárias e registros problemáticos."""
    df = df.copy()

    # Remover identificador sem valor preditivo
    df.drop(columns=["id"], inplace=True)

    # Remover o único registro com gender='Other' (impossível de generalizar)
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas com embasamento clínico."""
    df = df.copy()

    # Faixa etária (risco cresce com idade)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 40, 60, 100],
        labels=["child", "young_adult", "middle_age", "senior"],
    )

    # Glicose alta é fator de risco (> 140 mg/dL = hiperglicemia)
    df["high_glucose"] = (df["avg_glucose_level"] > 140).astype(int)

    # Obesidade (BMI > 30)
    df["obese"] = (df["bmi"] > 30).astype(int)

    # Combinação de fatores de risco cardiovascular
    df["cardio_risk_score"] = (
        df["hypertension"] + df["heart_disease"] + df["high_glucose"]
    )

    return df


# ─────────────────────────────────────────────
# 4. ENCODING DE CATEGÓRICAS
# ─────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica encoding adequado a cada variável categórica."""
    df = df.copy()

    # Binárias simples
    df["gender"] = (df["gender"] == "Male").astype(int)
    df["ever_married"] = (df["ever_married"] == "Yes").astype(int)
    df["Residence_type"] = (df["Residence_type"] == "Urban").astype(int)

    # Nominais com múltiplas categorias → One-Hot
    df = pd.get_dummies(df, columns=["work_type", "smoking_status", "age_group"], drop_first=False)

    # Converter booleanos gerados pelo get_dummies para int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


# ─────────────────────────────────────────────
# 5. TRANSFORMAÇÕES NUMÉRICAS
# ─────────────────────────────────────────────

def transform_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformação log em variáveis com alta assimetria."""
    df = df.copy()

    # avg_glucose_level tem skew positivo alto → log1p estabiliza
    df["avg_glucose_level"] = np.log1p(df["avg_glucose_level"])

    return df


# ─────────────────────────────────────────────
# 6. IMPUTAÇÃO DE NULOS
# ─────────────────────────────────────────────

def impute_missing(
    X_train: pd.DataFrame, X_test: pd.DataFrame, n_neighbors: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputa valores nulos usando KNN Imputer.
    Fit apenas no treino para evitar data leakage.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    cols = X_train.columns.tolist()

    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=cols)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=cols)

    return X_train_imp, X_test_imp


# ─────────────────────────────────────────────
# 7. ESCALONAMENTO
# ─────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Escala features numéricas contínuas com StandardScaler.
    Retorna também o scaler para uso posterior (ex: inferência).
    """
    continuous_cols = ["age", "avg_glucose_level", "bmi"]
    # Manter apenas as que existem no dataframe
    cols_to_scale = [c for c in continuous_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train, X_test, scaler


# ─────────────────────────────────────────────
# 8. BALANCEAMENTO COM SMOTE
# ─────────────────────────────────────────────

def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Aplica SMOTE apenas no conjunto de treino.
    NUNCA aplicar no teste — isso geraria data leakage.
    """
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"  Antes do SMOTE  → stroke=0: {(y_train==0).sum()}, stroke=1: {(y_train==1).sum()}")
    print(f"  Depois do SMOTE → stroke=0: {(y_res==0).sum()},  stroke=1: {(y_res==1).sum()}")

    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name="stroke")


# ─────────────────────────────────────────────
# 9. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def load_and_preprocess(
    path: Path = RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote_flag: bool = True,
    save_processed: bool = True,
) -> tuple:
    """
    Executa o pipeline completo de pré-processamento.

    Parâmetros
    ----------
    path : Path
        Caminho para o CSV. Padrão: data/raw/healthcare-dataset-stroke-data.csv
    test_size : float
        Proporção do conjunto de teste (padrão: 0.2).
    random_state : int
        Seed de reprodutibilidade.
    apply_smote_flag : bool
        Se True, aplica SMOTE no treino para balancear as classes.
    save_processed : bool
        Se True, salva X_train, X_test, y_train, y_test em data/processed/.

    Retorna
    -------
    X_train, X_test, y_train, y_test : DataFrames/Series prontos para modelagem.
    scaler : StandardScaler fitado (para uso em inferência futura).
    feature_names : lista de nomes das features após o processamento.
    """
    print("=" * 50)
    print("  PIPELINE DE PRÉ-PROCESSAMENTO")
    print("=" * 50)

    print("\n[1/8] Carregando dados...")
    df = load_data(path)
    print(f"  Origem: {Path(path)}")
    print(f"  Shape original: {df.shape}")

    print("\n[2/8] Limpando dados...")
    df = clean_data(df)
    print(f"  Shape após limpeza: {df.shape}")

    print("\n[3/8] Feature engineering...")
    df = feature_engineering(df)
    print(f"  Features criadas: age_group, high_glucose, obese, cardio_risk_score")

    print("\n[4/8] Encoding de categóricas...")
    df = encode_categoricals(df)

    print("\n[5/8] Transformações numéricas (log em glicose)...")
    df = transform_numerics(df)

    # Separar X e y antes do split
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    print("\n[6/8] Split treino/teste (estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    print("\n[7/8] Imputação de nulos (KNN Imputer, k=5)...")
    X_train, X_test = impute_missing(X_train, X_test)

    print("\n[7.5/8] Escalonamento de features contínuas...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    if apply_smote_flag:
        print("\n[8/8] Balanceamento com SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train, random_state)
    else:
        print("\n[8/8] SMOTE desativado — use class_weight no modelo.")

    feature_names = X_train.columns.tolist()

    print(f"\n✅ Pipeline concluído!")
    print(f"   Features finais: {len(feature_names)}")
    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")

    if save_processed:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(PROCESSED_DATA_PATH / "X_train.csv", index=False)
        X_test.to_csv(PROCESSED_DATA_PATH / "X_test.csv", index=False)
        y_train.to_csv(PROCESSED_DATA_PATH / "y_train.csv", index=False)
        y_test.to_csv(PROCESSED_DATA_PATH / "y_test.csv", index=False)
        print(f"   💾 Dados salvos em: {PROCESSED_DATA_PATH}")

    print("=" * 50)

    return X_train, X_test, y_train, y_test, scaler, feature_names


# ─────────────────────────────────────────────
# EXECUÇÃO DIRETA (teste rápido)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess()
    print("\nFeatures geradas:")
    for f in features:
        print(f"  • {f}")
