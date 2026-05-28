FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-api.txt pyproject.toml README.md ./
COPY stroke_prediction ./stroke_prediction

RUN pip install --upgrade pip && pip install -r requirements-api.txt

COPY reports ./reports

EXPOSE 8000

CMD ["uvicorn", "stroke_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]
