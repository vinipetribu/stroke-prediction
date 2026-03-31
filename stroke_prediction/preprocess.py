from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stroke_prediction.features import build_feature_matrix
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df["bmi"].dtype == object:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["id"], inplace=True)
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    return df


def run_preprocessing() -> None:
    print("[1/2] Carregando, limpando e aplicando feature engineering...")
    df = load_data()
    df = clean_data(df)
    df = build_feature_matrix(df)

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_PATH / "features.csv"
    df.to_csv(output_path, index=False)
    print(f"[2/2] Concluído! {df.shape[1]} colunas salvas em {output_path}")


if __name__ == "__main__":
    run_preprocessing()
