import logging
import os

import tensorflow as tf
from flask import Flask, jsonify, request

from sql_tokenizer import SQLTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MAX_WORDS = 10000
MAX_LEN = 100
VOCAB_PATH = os.getenv("VOCAB_PATH", "sql_tokenizer_vocab.json")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/sqli_model/3/")

sql_tokenizer = SQLTokenizer(max_words=MAX_WORDS, max_len=MAX_LEN)
sql_tokenizer.load_token_index(VOCAB_PATH)
logger.info("Loaded tokenizer vocabulary from %s (%d tokens)", VOCAB_PATH, len(sql_tokenizer.token_index))

loaded_model = tf.saved_model.load(MODEL_PATH)
model_predict = loaded_model.signatures["serving_default"]
logger.info("Loaded model from %s", MODEL_PATH)


def warm_up_model():
    """Sends a dummy request to the model to initialize it."""
    dummy_query = "SELECT * FROM users WHERE id = 1"
    query_seq = sql_tokenizer.texts_to_sequences([dummy_query])
    input_tensor = tf.convert_to_tensor(query_seq, dtype=tf.float32)
    _ = model_predict(input_tensor)
    logger.info("Model warmed up and ready to serve requests.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if not request.json or "query" not in request.json:
        return jsonify({"error": "No query provided"}), 400

    try:
        query = request.json["query"]
        query_seq = sql_tokenizer.texts_to_sequences([query])
        input_tensor = tf.convert_to_tensor(query_seq, dtype=tf.float32)

        prediction = model_predict(input_tensor)

        if "output_0" not in prediction or prediction["output_0"].get_shape() != [1, 1]:
            return jsonify({"error": "Invalid model output"}), 500

        confidence = float("%.4f" % prediction["output_0"].numpy()[0][0])
        return jsonify({"confidence": confidence})
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    warm_up_model()
    app.run(host="0.0.0.0", port=8000, debug=True)
