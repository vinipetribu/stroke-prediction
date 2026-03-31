"""
Engenharia de features para o dataset de AVC (stroke).

Fluxo esperado: DataFrame já limpo (sem `id`, sem gender Other) → engineer → encode → transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Colunas categóricas originais + derivadas que passam por one-hot
CATEGORICALS_FOR_OHE = [
    "work_type",
    "smoking_status",
    "age_group",
    "life_stage",
    "bmi_category",
    "glucose_category",
]

BINARY_MAP = [
    ("gender", "Male"),
    ("ever_married", "Yes"),
    ("Residence_type", "Urban"),
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas, bins clínicos, scores compostos e interações."""
    df = df.copy()

    # --- Idade ---
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 40, 60, 100],
        labels=["child", "young_adult", "middle_age", "senior"],
    )
    df["life_stage"] = pd.cut(
        df["age"],
        bins=[0, 35, 50, 65, 150],
        labels=["adult_young", "adult_mid", "pre_senior", "senior_plus"],
    )
    df["is_senior"] = (df["age"] >= 65).astype(np.int8)
    df["is_working_age"] = ((df["age"] >= 18) & (df["age"] < 65)).astype(np.int8)
    df["age_squared"] = (df["age"].astype(float) ** 2).astype(float)

    # --- IMC (WHO) ---
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[-np.inf, 18.5, 25, 30, np.inf],
        labels=["underweight", "normal", "overweight", "obese"],
    )
    df["bmi_category"] = df["bmi_category"].astype("string").fillna("missing")
    df["bmi_deficit"] = (df["bmi"] < 18.5).fillna(False).astype(np.int8)
    df["bmi_elevated"] = (df["bmi"] >= 25).fillna(False).astype(np.int8)

    # --- Glicemia (faixas tipo screening; avg_glucose_level no dataset) ---
    df["glucose_category"] = pd.cut(
        df["avg_glucose_level"],
        bins=[-np.inf, 100, 126, 140, np.inf],
        labels=["optimal", "elevated", "high", "very_high"],
    )
    df["glucose_category"] = df["glucose_category"].astype("string").fillna("missing")
    df["high_glucose"] = (df["avg_glucose_level"] > 140).astype(np.int8)
    df["prediabetes_range"] = (
        (df["avg_glucose_level"] >= 100) & (df["avg_glucose_level"] <= 125)
    ).astype(np.int8)

    # --- Scores compostos ---
    df["cardio_risk_score"] = (
        df["hypertension"] + df["heart_disease"] + df["high_glucose"]
    ).astype(np.int8)
    df["vascular_burden"] = (df["hypertension"] + df["heart_disease"]).astype(np.int8)
    df["metabolic_risk_count"] = (
        df["hypertension"]
        + (df["bmi"] >= 30).fillna(0).astype(int)
        + (df["avg_glucose_level"] > 100).astype(int)
    ).astype(np.int8)

    # --- Interações (antes de log em glicemia; árvores e regressão usam no treino) ---
    df["age_x_bmi"] = df["age"] * df["bmi"]
    df["age_x_glucose"] = df["age"] * df["avg_glucose_level"]
    df["glucose_x_bmi"] = df["avg_glucose_level"] * df["bmi"]
    df["hypertension_x_heart_disease"] = df["hypertension"] * df["heart_disease"]
    df["age_x_hypertension"] = df["age"] * df["hypertension"]
    df["bmi_x_hypertension"] = df["bmi"] * df["hypertension"]

    # Razão glicose / IMC (evita divisão por zero)
    df["glucose_per_bmi"] = df["avg_glucose_level"] / (df["bmi"].abs() + 1.0)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Binários explícitos + one-hot nas colunas categóricas listadas."""
    df = df.copy()

    for col, positive_value in BINARY_MAP:
        df[col] = (df[col] == positive_value).astype(np.int8)

    present_ohe = [c for c in CATEGORICALS_FOR_OHE if c in df.columns]
    df = pd.get_dummies(df, columns=present_ohe, drop_first=False, dtype=np.int8)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(np.int8)

    return df


def transform_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Suaviza caudas pesadas; mantém interações já calculadas em escala original."""
    df = df.copy()
    df["avg_glucose_level"] = np.log1p(df["avg_glucose_level"].clip(lower=0))
    # IMC com cauda à direita comum em dados reais
    bmi_pos = df["bmi"].clip(lower=0)
    df["bmi"] = np.log1p(bmi_pos)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de FE + encoding + transformações numéricas."""
    df = engineer_features(df)
    df = encode_categoricals(df)
    df = transform_numerics(df)
    return df
