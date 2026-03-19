# stroke-prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

---

## 🚀 Como configurar o projeto e baixar os dados (Para a Equipe)

Como estamos usando o **DVC** integrado ao **DagsHub** para versionar nossos dados e modelos separadamente do código, siga os passos abaixo para ter a base de dados completa na sua máquina:

### 1. Clone o repositório e instale as dependências

Primeiro, clone o repositório do Git e instale os pacotes necessários (recomenda-se o uso de um ambiente virtual):

```bash
git clone https://github.com/vinipetribu/stroke-prediction.git
cd stroke-prediction
pip install -r requirements.txt
```

### 2. Configure a autenticação do DagsHub

Para que o DVC consiga baixar os arquivos pesados (`.csv`, `.pkl`), você precisa estar autenticado no nosso remote do DagsHub:

1. Crie uma conta no [DagsHub](https://dagshub.com) e garanta que você tem acesso ao repositório `guiga-sa/stroke-prediction`.(https://dagshub.com/guiga-sa/stroke-prediction)
2. Gere um **token de acesso pessoal** no DagsHub indo em: **Settings → Tokens**.
3. No terminal do projeto, rode os comandos abaixo substituindo com os seus dados:

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user SEU_NOME_DE_USUARIO_DAGSHUB
dvc remote modify origin --local password SEU_TOKEN_DAGSHUB
```

> **Nota:** O uso da flag `--local` garante que suas credenciais fiquem salvas apenas na sua máquina e não sejam enviadas para o Git.

### 3. Baixe os dados (DVC Pull)

Com a autenticação pronta, basta rodar o comando abaixo para baixar as bases de dados e os modelos treinados:

```bash
dvc pull -r origin
```

Pronto! As pastas `data/raw/`, `data/processed/` e `models/` serão populadas com os arquivos corretos para a versão atual do código.

---

## 🔁 Reproduzindo o pipeline (DVC Repro)

Se você fizer alguma alteração no código de treino ou pré-processamento, pode reproduzir o pipeline inteiro com:

```bash
dvc repro
```

O DVC identificará automaticamente quais etapas foram afetadas pela sua mudança e executará apenas o necessário. Para forçar a re-execução de todas as etapas independentemente de mudanças, use:

```bash
dvc repro --force
```

Após reproduzir, não se esqueça de versionar os novos resultados:

```bash
dvc push -r origin
git add dvc.lock
git commit -m "repro: atualiza pipeline com novas alterações"
git push
```

---

## 📁 Organização do Projeto

```
├── LICENSE            <- Licença open-source do projeto
├── Makefile           <- Comandos úteis como `make data` ou `make train`
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
