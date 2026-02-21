FROM tensorflow/tensorflow:2.16.1

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
RUN poetry install --no-root

EXPOSE 8000

CMD poetry run gunicorn --bind ${HOST}:${PORT} --workers ${WORKERS} api:app
