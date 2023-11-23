# DeepSQLi

Deep learning model, dataset, trained model and related code for SQL injection detection.

## TensorFlow Serving

### Install TensorFlow Serving

```bash
pip install tensorflow-serving-api
```

### Start TensorFlow Serving

```bash
docker run -t --rm --name serving -p 8500-8501:8500-8501 -v /home/mostafa/gatewayd/DeepSQLi/sqli_model:/models/sqli_model -e MODEL_NAME=sqli_model tensorflow/serving
```

These logs should appear:

```bash
...
2023-11-23 23:04:43.350127: I tensorflow_serving/model_servers/server.cc:409] Running gRPC ModelServer at 0.0.0.0:8500 ...
...
2023-11-23 23:04:43.351796: I tensorflow_serving/model_servers/server.cc:430] Exporting HTTP/REST API at:localhost:8501 ...
```

### Start Tokenizer API

```bash
cd api
./run.sh
```

### Test

```bash
# Tokenize and sequence the query
curl 'http://localhost:5000/tokenize_and_sequence' -X POST -H 'Accept: application/json' -H 'Content-Type: application/json' --data-raw '{"query":"select * from users where id = 1 or 1=1"}'
# Predict whether the query is SQLi or not
curl 'http://localhost:8501/v1/models/sqli_model:predict' -X POST -H 'Accept: application/json' -H 'Content-Type: application/json' --data-raw '{"inputs":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,21,4,32,3,10,3,3]]}'
```
