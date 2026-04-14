# Guia do Projeto: Stroke Prediction

Este arquivo explica o funcionamento do projeto sem alterar o `README.md`.

O projeto é uma pipeline de Machine Learning/MLOps para prever risco de AVC (`stroke`) a partir de dados de saúde de pacientes. Ele pega um CSV bruto, faz limpeza e engenharia de features, treina alguns modelos de classificação e salva os modelos treinados junto com métricas de avaliação.

## Visão Geral

O fluxo principal é:

```text
data/raw/healthcare-dataset-stroke-data.csv
        |
        v
stroke_prediction/preprocess.py
        |
        v
data/processed/features.csv
        |
        v
stroke_prediction/train.py
        |
        v
models/*.pkl + reports/metrics.json
```

Ou seja:

1. O dado bruto fica em `data/raw/`.
2. O script `preprocess.py` limpa e transforma os dados.
3. O resultado vai para `data/processed/features.csv`.
4. O script `train.py` treina os modelos.
5. Os modelos treinados ficam em `models/`.
6. As métricas ficam em `reports/metrics.json`.

## Tecnologias Usadas

- `pandas`: leitura, limpeza e manipulação dos dados.
- `numpy`: operações numéricas.
- `scikit-learn`: modelos, treino, métricas e divisão treino/teste.
- `imbalanced-learn`: uso de `SMOTE` para lidar com desbalanceamento.
- `DVC`: versionamento de dados, modelos e pipeline.
- `MLflow`: rastreamento opcional de experimentos.
- `ruff`: lint e formatação de código.

## Dataset

O dataset bruto fica em:

```text
data/raw/healthcare-dataset-stroke-data.csv
```

Ele tem 5.110 linhas. Algumas colunas importantes:

- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `ever_married`
- `work_type`
- `Residence_type`
- `avg_glucose_level`
- `bmi`
- `smoking_status`
- `stroke`

A variável alvo é:

```text
stroke
```

Valores:

- `0`: paciente não teve AVC
- `1`: paciente teve AVC

Esse é um problema de classificação desbalanceada, porque existem muito mais casos `stroke = 0` do que `stroke = 1`.

## Estrutura do Projeto

```text
data/raw/                Dados brutos, versionados pelo DVC
data/processed/          Dados processados para treino
models/                  Modelos treinados
notebooks/               Notebooks de exploração
reports/                 Métricas e relatórios
stroke_prediction/       Código Python principal
dvc.yaml                 Definição da pipeline DVC
dvc.lock                 Versões/checksums dos artefatos gerados
requirements.txt         Dependências Python
pyproject.toml           Configurações do projeto
Makefile                 Comandos utilitários
```

Arquivos mais importantes:

- `stroke_prediction/preprocess.py`
- `stroke_prediction/features.py`
- `stroke_prediction/train.py`
- `dvc.yaml`
- `reports/metrics.json`

Arquivos que ainda parecem ser templates do Cookiecutter:

- `stroke_prediction/dataset.py`
- `stroke_prediction/plots.py`
- `stroke_prediction/modeling/predict.py`

## Pipeline DVC

A pipeline está definida em:

```text
dvc.yaml
```

Ela possui duas etapas principais.

## Etapa 1: Preprocessamento

Comando:

```bash
python stroke_prediction/preprocess.py
```

Entradas rastreadas pelo DVC:

```text
data/raw/healthcare-dataset-stroke-data.csv
stroke_prediction/preprocess.py
```

Saída:

```text
data/processed/features.csv
```

O que acontece nessa etapa:

1. O CSV bruto é carregado.
2. A coluna `bmi` é convertida para número.
3. Valores inválidos como `N/A` viram valores ausentes.
4. A coluna `id` é removida.
5. Linhas com `gender = Other` são removidas.
6. A função `build_feature_matrix` é chamada.
7. O arquivo final é salvo em `data/processed/features.csv`.

Importante: `preprocess.py` usa funções de `features.py`, mas `features.py` não está listado atualmente como dependência no `dvc.yaml`. Isso significa que alterações em `features.py` podem não disparar automaticamente o `dvc repro`.

## Etapa 2: Treinamento

Comando:

```bash
python stroke_prediction/train.py
```

Entrada:

```text
data/processed/features.csv
```

Saídas:

```text
models/logistic_regression.pkl
models/random_forest.pkl
models/hist_gradient_boosting.pkl
reports/metrics.json
```

O que acontece nessa etapa:

1. O arquivo `features.csv` é carregado.
2. A coluna `stroke` é separada como alvo.
3. Os dados são divididos em treino e teste.
4. Três modelos são treinados.
5. Cada modelo é avaliado com `classification_report`.
6. Os modelos são salvos em `.pkl`.
7. As métricas são salvas em `reports/metrics.json`.

A divisão treino/teste usa:

```text
test_size = 0.2
random_state = 42
stratify = y
```

O `stratify=y` é importante porque mantém a proporção de classes entre treino e teste.

## Engenharia de Features

A engenharia de features está em:

```text
stroke_prediction/features.py
```

O fluxo interno é:

```text
build_feature_matrix
    -> engineer_features
    -> encode_categoricals
    -> transform_numerics
```

### Features de idade

- `age_group`
- `life_stage`
- `is_senior`
- `is_working_age`
- `age_squared`

### Features de IMC

- `bmi_category`
- `bmi_deficit`
- `bmi_elevated`

### Features de glicose

- `glucose_category`
- `high_glucose`
- `prediabetes_range`

### Scores de risco

- `cardio_risk_score`
- `vascular_burden`
- `metabolic_risk_count`

### Interações

- `age_x_bmi`
- `age_x_glucose`
- `glucose_x_bmi`
- `hypertension_x_heart_disease`
- `age_x_hypertension`
- `bmi_x_hypertension`
- `glucose_per_bmi`

### Encoding

Algumas colunas viram binárias:

- `gender`
- `ever_married`
- `Residence_type`

Outras colunas categóricas passam por one-hot encoding:

- `work_type`
- `smoking_status`
- `age_group`
- `life_stage`
- `bmi_category`
- `glucose_category`

### Transformações numéricas

As colunas abaixo recebem `log1p`:

- `avg_glucose_level`
- `bmi`

Isso ajuda a suavizar distribuições com cauda longa.

## Modelos Treinados

O arquivo `train.py` treina três modelos.

## 1. Logistic Regression

Pipeline:

```text
KNNImputer -> StandardScaler -> SMOTE -> LogisticRegression
```

Usa `StandardScaler` porque regressão logística é sensível à escala das variáveis.

## 2. Random Forest

Pipeline:

```text
KNNImputer -> SMOTE -> RandomForestClassifier
```

Não usa scaler porque modelos de árvore não precisam de padronização.

## 3. HistGradientBoosting

Pipeline:

```text
KNNImputer -> SMOTE -> HistGradientBoostingClassifier
```

É um modelo de gradient boosting do scikit-learn.

## Por Que SMOTE É Usado?

O dataset é desbalanceado. Existem poucos casos de AVC em comparação com casos sem AVC.

Sem tratar isso, o modelo pode aprender a prever quase sempre `0` e ainda assim ter uma acurácia alta. Isso seria ruim, porque o objetivo é detectar casos de risco.

O `SMOTE` cria exemplos sintéticos da classe minoritária durante o treino.

Neste projeto, o SMOTE está dentro da pipeline de treino. Isso é bom porque ele é aplicado depois da divisão treino/teste, evitando contaminar o conjunto de teste.

## Métricas Atuais

As métricas ficam em:

```text
reports/metrics.json
```

Resultados locais atuais:

| Modelo | Accuracy | Recall classe 1 | Precision classe 1 | F1 classe 1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.743 | 0.760 | 0.131 | 0.224 |
| Random Forest | 0.944 | 0.000 | 0.000 | 0.000 |
| HistGradientBoosting | 0.948 | 0.100 | 0.385 | 0.159 |

O ponto principal: acurácia pode enganar.

O Random Forest tem alta acurácia, mas não detectou nenhum caso positivo no teste. Para um problema de AVC, isso é perigoso, porque perder casos positivos pode ser pior do que gerar falsos positivos.

Neste momento, a Regressão Logística parece ser o melhor baseline para detectar AVC, porque tem recall maior para a classe `1`.

## Como Rodar o Projeto

Instalar dependências:

```bash
python -m pip install -r requirements.txt
```

Baixar artefatos com DVC:

```bash
dvc pull -r origin
```

Rodar a pipeline completa:

```bash
dvc repro
```

Forçar a execução de tudo:

```bash
dvc repro --force
```

Rodar manualmente:

```bash
python stroke_prediction/preprocess.py
python stroke_prediction/train.py
```

## Como Rodar o MLflow

Use dois terminais.

Terminal 1:

```bash
mlflow server --host 127.0.0.1 --port 5002
```

Abra no navegador:

```text
http://127.0.0.1:5002
```

Terminal 2:

```bash
python stroke_prediction/train.py
```

O script `train.py` usa por padrão:

```text
MLFLOW_TRACKING_URI=http://127.0.0.1:5002
MLFLOW_EXPERIMENT_NAME=stroke-prediction
```

Para desativar MLflow:

```bash
MLFLOW_DISABLE=1 python stroke_prediction/train.py
```

## Comandos Uteis

```bash
make requirements
make lint
make format
make clean
dvc repro
dvc push -r origin
```

## O Que Voce Precisa Saber

Resumo essencial:

1. A pipeline real é `preprocess.py` -> `train.py`.
2. O DVC controla a reprodutibilidade com `dvc.yaml` e `dvc.lock`.
3. A maior parte da lógica de ML está em `features.py` e `train.py`.
4. O problema é desbalanceado, então acurácia sozinha não basta.
5. Para esse caso, recall da classe `1` é muito importante.
6. O modelo com maior acurácia não é necessariamente o melhor modelo.
7. Ainda não existe uma API ou script real de inferência pronto.

## Melhorias Recomendadas

1. Adicionar `stroke_prediction/features.py` como dependência do estágio `preprocess` no `dvc.yaml`.
2. Implementar `stroke_prediction/modeling/predict.py`.
3. Fazer tuning de threshold para a classe `1`.
4. Usar validação cruzada.
5. Criar testes para preprocessing e feature engineering.
6. Melhorar o tracking no MLflow.
7. Criar uma API com FastAPI para servir predições.
