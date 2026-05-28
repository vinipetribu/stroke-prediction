#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = stroke-prediction
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python3
VENV = .venv
DAGSHUB_REPO_OWNER = guiga-sa
DAGSHUB_REPO_NAME = stroke-prediction
DAGSHUB_URL = https://dagshub.com/$(DAGSHUB_REPO_OWNER)/$(DAGSHUB_REPO_NAME)

ifeq ($(OS),Windows_NT)
	VENV_BIN = $(VENV)/Scripts
else
	VENV_BIN = $(VENV)/bin
endif

-include .env
export

#################################################################################
# SETUP                                                                         #
#################################################################################

.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv $(VENV)
	@echo ">>> Ambiente virtual criado em .venv"
	@echo ">>> Ative com: source .venv/bin/activate  (Linux/Mac)"
	@echo "               .venv\\Scripts\\activate     (Windows)"

.PHONY: requirements
requirements:
	$(VENV_BIN)/pip install -U pip
	$(VENV_BIN)/pip install -r requirements.txt
	@echo ">>> Dependências instaladas."

.PHONY: dagshub_config
dagshub_config:
	@if [ -z "$(DAGSHUB_TOKEN)" ]; then \
		echo "ERRO: DAGSHUB_TOKEN não definido no .env"; exit 1; \
	fi
	$(VENV_BIN)/dvc remote add -d -f dagshub $(DAGSHUB_URL).dvc
	$(VENV_BIN)/dvc remote modify dagshub --local auth basic
	$(VENV_BIN)/dvc remote modify dagshub --local user $(DAGSHUB_USER)
	$(VENV_BIN)/dvc remote modify dagshub --local password $(DAGSHUB_TOKEN)
	@echo ">>> DVC configurado com DagsHub."

.PHONY: setup
setup: create_environment requirements dagshub_config
	@echo ""
	@echo "========================================="
	@echo " Projeto configurado com sucesso!"
	@echo " Certifique-se de preencher o .env com:"
	@echo "   DAGSHUB_USER=seu_usuario"
	@echo "   DAGSHUB_TOKEN=seu_token"
	@echo "========================================="

#################################################################################
# DVC                                                                           #
#################################################################################

.PHONY: dvc_pull
dvc_pull:
	$(VENV_BIN)/dvc pull

.PHONY: dvc_push
dvc_push:
	$(VENV_BIN)/dvc push

.PHONY: run
run:
	$(VENV_BIN)/dvc repro

.PHONY: api
api:
	MLFLOW_TRACKING_URI=http://127.0.0.1:5002 MLFLOW_EXPERIMENT_NAME=stroke-prediction-compose $(VENV_BIN)/uvicorn stroke_prediction.api:app --host 0.0.0.0 --port 8000 --reload

.PHONY: experiments
experiments:
	PATH=$(VENV_BIN):$$PATH MLFLOW_TRACKING_URI=http://127.0.0.1:5002 MLFLOW_EXPERIMENT_NAME=stroke-prediction-compose $(VENV_BIN)/dvc repro --force train

.PHONY: activity
activity:
	docker compose up --build -d mlflow
	sleep 5
	PATH=$(VENV_BIN):$$PATH MLFLOW_TRACKING_URI=http://127.0.0.1:5002 MLFLOW_EXPERIMENT_NAME=stroke-prediction-compose $(VENV_BIN)/dvc repro --force train
	docker compose up --build -d api

.PHONY: compose_up
compose_up:
	docker compose up --build

.PHONY: compose_down
compose_down:
	docker compose down

#################################################################################
# PROJETO                                                                       #
#################################################################################

.PHONY: data
data:
	$(VENV_BIN)/python stroke_prediction/dataset.py

#################################################################################
# QUALIDADE DE CÓDIGO                                                           #
#################################################################################

.PHONY: lint
lint:
	$(VENV_BIN)/ruff format --check
	$(VENV_BIN)/ruff check

.PHONY: format
format:
	$(VENV_BIN)/ruff check --fix
	$(VENV_BIN)/ruff format

.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	@echo ">>> Arquivos temporários removidos."

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
