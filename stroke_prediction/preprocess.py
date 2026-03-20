## Imports das libs usadas

from pathlib import Path
import numpy as np 
import pandas as pd 

## Definindo paths por meio das ferramentas que importei da lib pathlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]#path raiz
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv" #dados sujos
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" #dados processados

#DEFINIÇÃO DAS FUNÇÕES

#CARREGANDO DADOS

def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame: #espera receber path e retorna algo do tipo DF
    df = pd.read_csv(path)
    if df["bmi"].dtype == object: #tratamento preliminar coluna BMI ajuste de tipo
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame: #recebe df retorna df :)
    df = df.copy() #cria copia do df por segurança para modificar ele sem erro
    df.drop(columns=["id"], inplace= True)#remove coluna id
    df = df[df["gender"] != "Other"].reset_index(drop=True)# remove linhas com genero other, acho que eram 2
    return df
 
 
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"], 
        bins = [0, 18, 40, 60, 100],
        labels=["child", "young_adult", "middle_age", "senior"]
    )#agrupando e catalogando as pessoas com base nos grupos de idade 
    df["high_glucose"] = (df["avg_glucose_level"]>140).astype(int)#criação de coluna binaria nova classifcando
    df["obese"] = (df["bmi"]>30).astype(int)
    df["cardio_risk_score"] = df["hypertension"] + df["heart_disease"] + df["high_glucose"]
    
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = (df["gender"] == "Male").astype(int)#tranformando em colunas binarias numericas
    df["ever_married"] = (df["ever_married"] == "Yes").astype(int)
    df["Residence_type"] = (df["Residence_type"] == "Urban").astype(int)
    
    df = pd.get_dummies(df, columns=["work_type", "smoking_status", "age_group"], drop_first=False)##"onehotencoding" de certa forma kkk
    bool_cols = df.select_dtypes(include="bool").columns#selecionando colunas booleanas e transformando em numerica
    df[bool_cols] = df[bool_cols].astype(int)
    return df

def transform_numerics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_glucose_level"] = np.log1p(df["avg_glucose_level"])#aplicando log para suavizar tudo 
    return df

def run_preprocessing():#fluxo de execução e processamento de dados, salvando eles no fim
    print("[1/2] Executando pré-processamento estático...")
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_categoricals(df)
    df = transform_numerics(df)
    
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_PATH / "features.csv"
    df.to_csv(output_path, index=False)
    print(f"[2/2] Concluído! Dados salvos em {output_path}")

if __name__ == "__main__":
    run_preprocessing()