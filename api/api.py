import time

import pandas as pd
from flask import Flask, jsonify, request
from flask_caching import Cache
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

MAX_WORDS = 10000
MAX_LEN = 100
DATASET = pd.read_csv("../dataset/sqli_dataset.csv")
CONFIG = {"DEBUG": True, "CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300}


def get_query_vec(query):
    # Tokenize the sample
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(DATASET["Query"])

    # Vectorize the sample
    query_seq = tokenizer.texts_to_sequences([query])
    query_vec = pad_sequences(query_seq, maxlen=MAX_LEN)

    return query_vec.tolist()


def create_app():
    app = Flask(__name__)
    app.config.from_mapping(CONFIG)
    cache = Cache(app)

    @app.route("/tokenize_and_sequence/<query>", methods=["GET"])
    @cache.cached(timeout=300)
    def tokenize_and_sequence(query):
        now = time.time()
        tokens = get_query_vec(query)
        print("Time taken (s): ", time.time() - now)
        return jsonify({"tokens": tokens[0]})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="localhost", port=5000, debug=True)
