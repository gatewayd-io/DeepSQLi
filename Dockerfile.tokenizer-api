FROM tensorflow/tensorflow:latest

ENV KMP_AFFINITY=noverbose
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV DATASET_PATH=/app/sqli_dataset.csv
ENV WORKERS=4
ENV HOST=0.0.0.0
ENV PORT=8000

WORKDIR /app
COPY api/api.py /app
COPY api/pyproject.toml /app
COPY api/poetry.lock /app
COPY dataset/sqli_dataset.csv /app
RUN pip install --disable-pip-version-check poetry
RUN poetry install --no-root

EXPOSE 8000

CMD poetry run gunicorn --bind ${HOST}:${PORT} --workers ${WORKERS} api:app
