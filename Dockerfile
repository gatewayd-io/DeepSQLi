# NOTE: upstream used tensorflow/tensorflow:2.16.1, which ships Python 3.11 and
# is rejected by pyproject.toml's `python = "^3.12"` constraint. Build on a
# Python 3.12 base instead and let Poetry install TensorFlow from poetry.lock.
FROM python:3.12-slim-bookworm

# libgomp1 is required at runtime by the TensorFlow CPU wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV KMP_AFFINITY=noverbose
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV VOCAB_PATH=/app/sql_tokenizer_vocab.json
ENV MODEL_PATH=/app/sqli_model/3/
ENV WORKERS=4
ENV HOST=0.0.0.0
ENV PORT=8000

WORKDIR /app
COPY api/api.py /app/
COPY api/pyproject.toml /app/
COPY api/poetry.lock /app/
COPY training/sql_tokenizer.py /app/
COPY training/sql_tokenizer_vocab.json /app/
COPY sqli_model/3/ /app/sqli_model/3/
RUN pip install --disable-pip-version-check poetry
# Committed poetry.lock is out of sync with pyproject.toml; refresh it, then install.
RUN poetry lock
RUN poetry install --no-root

EXPOSE 8000

CMD poetry run gunicorn --bind ${HOST}:${PORT} --workers ${WORKERS} api:app
