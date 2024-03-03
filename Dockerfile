FROM tensorflow/tensorflow:latest-py3

ENV KMP_AFFINITY=noverbose
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV DATASET_PATH=/app/sqli_dataset.csv
ENV WORKERS=4
ENV HOST=0.0.0.0
ENV PORT=8000

WORKDIR /app
COPY api/api.py /app
COPY api/requirements.txt /app
COPY dataset/sqli_dataset.csv /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "-b", "${HOST}:${PORT}", "-w", ${WORKERS}, "api:app"]
