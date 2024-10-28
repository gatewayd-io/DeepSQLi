import sys
from pathlib import Path

from tensorflow import float32
from tensorflow.io import TFRecordWriter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
import pandas as pd

# Check if the input and output model directories are provided
if len(sys.argv) != 4:
    print("Usage: python train.py <dataset_file> <model_dir> <is_keras_tensor>")
    sys.exit(1)

# Define parameters and constants
MAX_WORDS = 10000
MAX_LEN = 100
WARM_UP_FILE = "tf_serving_warmup_requests"
WARM_UP_DIR = Path(sys.argv[2]) / "assets.extra/"
NUM_RECORDS = 1

# Load dataset
data = pd.read_csv(sys.argv[1])

# Use Tokenizer to encode text
tokenizer = Tokenizer(num_words=MAX_WORDS, filters="")
tokenizer.fit_on_texts(data["Query"])
sequences = tokenizer.texts_to_sequences(data["Query"])

# Tokenize and sequence the query
query_seq = tokenizer.texts_to_sequences(
    [
        "SELECT * FROM users WHERE username = 'admin' AND password = 'password' or '1'='1';"
    ]
)
query_vec = pad_sequences(query_seq, maxlen=MAX_LEN)

# Create directory for warm-up requests if it does not exist
WARM_UP_DIR.mkdir(parents=True, exist_ok=True)

# Write warm-up requests to a TFRecord file
with TFRecordWriter(str(WARM_UP_DIR / WARM_UP_FILE)) as writer:
    # Create a prediction request
    predict_request = predict_pb2.PredictRequest()
    predict_request.model_spec.name = "sqli_model"
    predict_request.model_spec.signature_name = "serving_default"
    if sys.argv[3].lower() == "true":
        predict_request.inputs["keras_tensor_32"].CopyFrom(
            tensor_util.make_tensor_proto(query_vec, float32)
        )
    else:
        predict_request.inputs["embedding_input"].CopyFrom(
            tensor_util.make_tensor_proto(query_vec, float32)
        )

    # Create a prediction log
    prediction_log = prediction_log_pb2.PredictionLog(
        predict_log=prediction_log_pb2.PredictLog(request=predict_request)
    )

    for r in range(NUM_RECORDS):
        # Write the prediction log to a TFRecord file
        writer.write(prediction_log.SerializeToString())
