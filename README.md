# DeepSQLi

Deep learning model, dataset, trained model and related code for SQL injection detection.

## Docker

```bash
# Build the images
docker build --no-cache --tag tokenizer-api:latest -f Dockerfile.tokenizer-api .
docker build --no-cache --tag serving-api:latest -f Dockerfile.serving-api .
# Run the Tokenizer and Serving API containers
docker run --rm --name tokenizer-api -p 8000:8000 -d tokenizer-api:latest
docker run --rm --name serving-api -p 8500-8501:8500-8501 -d serving-api:latest
```

## Docker Compose

```bash
# Run the Tokenizer and Serving API containers
docker compose up -d
# Stop the Tokenizer and Serving API containers
docker compose down
```

### Test

```bash
# Tokenize and sequence the query
curl 'http://localhost:8000/tokenize_and_sequence' -X POST -H 'Accept: application/json' -H 'Content-Type: application/json' --data-raw '{"query":"select * from users where id = 1 or 1=1"}'
# Predict whether the query is SQLi or not
curl 'http://localhost:8501/v1/models/sqli_model:predict' -X POST -H 'Accept: application/json' -H 'Content-Type: application/json' --data-raw '{"inputs":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,21,4,32,3,10,3,3]]}'

# Or you can use the following one-liner:
curl -s 'http://localhost:8501/v1/models/sqli_model:predict' -X POST -H 'Accept: application/json' -H 'Content-Type: application/json' --data-raw '{"inputs":['$(curl -s 'http://localhost:8000/tokenize_and_sequence' -X POST -H 'Accept: application/json' -H 'Content-Type: application/json' --data-raw '{"query":"select * from users where id = 1 or 1=1"}' | jq -c .tokens)']}' | jq
```
