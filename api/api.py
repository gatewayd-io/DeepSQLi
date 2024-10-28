from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import os

app = Flask(__name__)

# Constants and configurations
MAX_WORDS = 10000
MAX_LEN = 100
DATASET_PATH = os.getenv("DATASET_PATH", "dataset/sqli_dataset1.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/sqli_model/3/")
DATASET = pd.read_csv(DATASET_PATH)

# Tokenizer setup
TOKENIZER = Tokenizer(num_words=MAX_WORDS, filters="")
TOKENIZER.fit_on_texts(DATASET["Query"])

# Load the model using tf.saved_model.load and get the serving signature
loaded_model = tf.saved_model.load(MODEL_PATH)
model_predict = loaded_model.signatures["serving_default"]


@app.route("/predict", methods=["POST"])
def predict():
    if not request.json or "query" not in request.json:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Tokenize and pad the input query
        query = request.json["query"]
        query_seq = TOKENIZER.texts_to_sequences([query])
        query_vec = pad_sequences(query_seq, maxlen=MAX_LEN)

        # Convert input to tensor
        input_tensor = tf.convert_to_tensor(query_vec, dtype=tf.float32)

        # Use the loaded model's serving signature to make the prediction
        prediction = model_predict(input_tensor)

        if "output_0" not in prediction or prediction["output_0"].get_shape() != [1, 1]:
            return jsonify({"error": "Invalid model output"}), 500

        return jsonify(
            {
                "confidence": float("%.4f" % prediction["output_0"].numpy()[0][0]),
            }
        )
    except Exception as e:
        # TODO: Log the error and return a proper error message
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
