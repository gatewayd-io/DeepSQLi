# DeepSQLi

Deep learning model, dataset, trained model and related code for SQL injection detection.

## Docker

```bash
# Build the image
docker build --no-cache --tag deepsqli-api:latest .
# Run the DeepSQLi API container
docker run --rm --name deepsqli-api -p 8000:8000 -d deepsqli-api:latest
# Run the TensorFlow Serving API container
docker run -t --rm --name serving-api -p 8500-8501:8500-8501 -v ./sqli_model:/models/sqli_model -e MODEL_NAME=sqli_model tensorflow/serving
```

### Test

```bash
# Tokenize and sequence the query
curl 'http://localhost:8000/tokenize_and_sequence' -X POST -H 'Accept: application/json' -H 'Content-Type: application/json' --data-raw '{"query":"select * from users where id = 1 or 1=1"}'
# Predict whether the query is SQLi or not
curl 'http://localhost:8501/v1/models/sqli_model:predict' -X POST -H 'Accept: application/json' -H 'Content-Type: application/json' --data-raw '{"inputs":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,21,4,32,3,10,3,3]]}'

# Or you can use the following one-liner:
TOKENS=$(curl -s 'http://localhost:8000/tokenize_and_sequence' -X POST -H 'Accept: application/json' -H 'Content-Type: application/json' --data-raw '{"query":"select * from users where id = 1 or 1=1"}' | jq -c .tokens) | curl -s 'http://localhost:8501/v1/models/sqli_model:predict' -X POST -H 'Accept: application/json' -H 'Content-Type: application/json' --data-raw '{"inputs":['${TOKENS}']}' | jq
```
