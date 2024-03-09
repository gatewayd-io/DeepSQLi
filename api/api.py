import os

import pandas as pd
from flask import Flask, jsonify, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

MAX_WORDS = 10000
MAX_LEN = 100
DATASET_PATH = os.getenv("DATASET_PATH", "dataset/sqli_dataset.csv")
DATASET = pd.read_csv(DATASET_PATH)
TOKENIZER = Tokenizer(num_words=MAX_WORDS, filters="")
TOKENIZER.fit_on_texts(DATASET["Query"])
CONFIG = {"DEBUG": False}


app = Flask(__name__)
app.config.from_mapping(CONFIG)


@app.route("/tokenize_and_sequence", methods=["POST"])
def tokenize_and_sequence():
    """Tokenize and sequence the input query from the request
    and return the vectorized output.
    """

    body = request.get_json()
    if not body:
        return jsonify({"error": "No JSON body provided"}), 400

    # Vectorize the sample
    query_seq = TOKENIZER.texts_to_sequences([body["query"]])
    query_vec = pad_sequences(query_seq, maxlen=MAX_LEN)

    tokens = query_vec.tolist()
    return jsonify({"tokens": tokens[0]})


if __name__ == "__main__":
    # Run the app in debug mode
    app.run(host="localhost", port=8000, debug=True)
