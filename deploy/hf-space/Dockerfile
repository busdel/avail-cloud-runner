FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy pandas scikit-learn scipy statsmodels matplotlib seaborn \
        openpyxl networkx xgboost flask waitress jinja2 markdown

COPY app/ /app/

ENV AVAIL_WEB_ROOT=/app \
    AVAIL_WEB_PORT=8090 \
    AVAIL_ALLOW_SERVER_RUN=0 \
    AVAIL_MAX_UPLOAD_MB=2048 \
    AVAIL_PYTHON=python3 \
    PYTHONPATH=/app:/app/avail_ai

EXPOSE 8090
CMD ["bash", "entrypoint.sh"]
