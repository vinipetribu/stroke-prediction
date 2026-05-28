# stroke-prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

---

## 🚀 Como rodar o projeto do zero com Makefile

Como estamos usando o **DVC** integrado ao **DagsHub** para versionar dados e modelos separadamente do código, o fluxo recomendado é usar os comandos do `Makefile`.

### 1. Clone o repositório

```bash
git clone https://github.com/vinipetribu/stroke-prediction.git
cd stroke-prediction
```

### 2. Configure as credenciais do DagsHub

Crie uma conta no [DagsHub](https://dagshub.com) e garanta que você tem acesso ao repositório [`guiga-sa/stroke-prediction`](https://dagshub.com/guiga-sa/stroke-prediction).

Depois, gere um **token de acesso pessoal** no DagsHub em **Settings → Tokens**.

Na raiz do projeto, crie um arquivo `.env` com suas credenciais:

```bash
DAGSHUB_USER=seu_usuario
DAGSHUB_TOKEN=seu_token
```

> **Nota:** O `.env` não deve ser enviado para o Git. Ele é lido automaticamente pelo `Makefile` para configurar o DVC localmente.

### 3. Configure o ambiente do projeto

Rode:

```bash
make setup
```

Esse comando executa, em sequência:

- `make create_environment`: cria o ambiente virtual em `.venv`;
- `make requirements`: instala as dependências do `requirements.txt`;
- `make dagshub_config`: configura o remote `dagshub` do DVC usando as credenciais do `.env`.

### 4. Baixe os dados e modelos versionados

Com o ambiente configurado, baixe os arquivos controlados pelo DVC:

```bash
make dvc_pull
```

As pastas `data/raw/`, `data/processed/` e `models/` serão populadas com os arquivos corretos para a versão atual do código.

### 5. Rode o pipeline

Para reproduzir o pipeline configurado no DVC:

```bash
make run
```

Esse comando executa `dvc repro` usando o ambiente virtual criado pelo projeto.

### Comandos úteis

```bash
make help       # Lista os comandos disponíveis
make data       # Executa o script stroke_prediction/dataset.py
make api        # Sobe a API FastAPI localmente em http://localhost:8000
make experiments # Roda os experimentos e registra modelos no MLflow local
make activity   # Sobe MLflow, roda experimentos e sobe a API via Docker
make compose_up # Sobe MLflow + API FastAPI com Docker Compose
make lint       # Verifica formatação e lint com ruff
make format     # Corrige lint e formata o código com ruff
make dvc_push   # Envia dados/modelos versionados para o remote DVC
make clean      # Remove arquivos temporários do Python
```

---

## 🧪 MLflow, FastAPI e modelo champion

Esta entrega adiciona dois serviços no `docker-compose.yml`:

- `mlflow`: servidor MLflow em `http://localhost:5002`, com banco SQLite e artefatos persistidos no volume Docker `mlflow-data`.
- `api`: API FastAPI em `http://localhost:8000`, carregando o modelo champion pelo MLflow.

Suba a stack:

```bash
make compose_up
```

Em outro terminal, rode os experimentos para registrar os modelos no MLflow, marcar o alias `champion` no Model Registry e atualizar `reports/champion.json` com a URI do champion:

```bash
make experiments
```

Para fazer o fluxo da atividade em um único comando:

```bash
make activity
```

Endpoints principais:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```

Exemplo de predição:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

### Champion escolhido

O modelo champion é `logistic_regression`. A API não carrega esse modelo pela pasta local `models/`; ela usa a URI `mlflow_model_uri` registrada em `reports/champion.json`. Quando os experimentos rodam com MLflow ativo, essa URI fica no formato `models:/stroke-prediction-champion@champion`.

A decisão está registrada em `reports/champion.json`. Apesar de `hist_gradient_boosting` ter a maior acurácia (`0.9481`), o dataset é desbalanceado e a atividade envolve triagem de AVC. Por isso, o critério escolhido foi o recall da classe positiva (`stroke=1`):

- `logistic_regression`: recall `0.76`
- `hist_gradient_boosting`: recall `0.10`
- `random_forest`: recall `0.00`

---

## 🔁 Reproduzindo o pipeline (DVC Repro)

Se você fizer alguma alteração no código de treino ou pré-processamento, pode reproduzir o pipeline inteiro com:

```bash
make run
```

O comando executa `dvc repro`. O DVC identificará automaticamente quais etapas foram afetadas pela sua mudança e executará apenas o necessário.

Para forçar a re-execução de todas as etapas independentemente de mudanças, ative o ambiente virtual e rode o DVC diretamente:

```bash
source .venv/bin/activate
dvc repro --force
```

Após reproduzir, não se esqueça de versionar os novos resultados:

```bash
make dvc_push
git add dvc.lock
git commit -m "repro: atualiza pipeline com novas alterações"
git push
```

---

## 📁 Organização do Projeto

```
├── LICENSE            <- Licença open-source do projeto
├── Makefile           <- Comandos úteis como `make setup`, `make run` ou `make data`
├── README.md          <- README principal para os desenvolvedores
├── data
│   ├── external       <- Dados de fontes externas
│   ├── interim        <- Dados intermediários transformados
│   ├── processed      <- Dados finais prontos para modelagem
│   └── raw            <- Dados originais, imutáveis
│
├── docs               <- Projeto mkdocs padrão; veja www.mkdocs.org
│
├── models             <- Modelos treinados e serializados, predições ou resumos
│
├── notebooks          <- Jupyter notebooks. Convenção de nomes: número (para ordenação),
│                         iniciais do autor e descrição curta, ex:
│                         `1.0-jqp-initial-data-exploration`
│
├── pyproject.toml     <- Configuração do projeto com metadados do pacote
│                         stroke_prediction e configuração de ferramentas como black
│
├── references         <- Dicionários de dados, manuais e outros materiais explicativos
│
├── reports            <- Análises geradas em HTML, PDF, LaTeX, etc.
│   └── figures        <- Gráficos e figuras gerados para relatórios
│
├── requirements.txt   <- Arquivo de dependências para reproduzir o ambiente, ex:
│                         gerado com `pip freeze > requirements.txt`
│
├── setup.cfg          <- Arquivo de configuração para o flake8
│
└── stroke_prediction  <- Código-fonte do projeto
    │
    ├── __init__.py             <- Torna stroke_prediction um módulo Python
    │
    ├── config.py               <- Variáveis úteis e configurações
    │
    ├── dataset.py              <- Scripts para baixar ou gerar dados
    │
    ├── features.py             <- Código para criar features para modelagem
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Código para inferência com modelos treinados
    │   └── train.py            <- Código para treinar modelos
    │
    └── plots.py                <- Código para criar visualizações
```
