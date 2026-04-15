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
make lint       # Verifica formatação e lint com ruff
make format     # Corrige lint e formata o código com ruff
make dvc_push   # Envia dados/modelos versionados para o remote DVC
make clean      # Remove arquivos temporários do Python
```

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
