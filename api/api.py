from flask import Flask, jsonify, request
import tensorflow as tf
import pandas as pd
import os
from sql_tokenizer import SQLTokenizer  # Import SQLTokenizer

app = Flask(__name__)

# Constants and configurations
MAX_WORDS = 10000
MAX_LEN = 100
DATASET_PATH = os.getenv("DATASET_PATH", "dataset/sqli_dataset1.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/sqli_model/3/")

# Load dataset and initialize SQLTokenizer
DATASET = pd.read_csv(DATASET_PATH)
sql_tokenizer = SQLTokenizer(max_words=MAX_WORDS, max_len=MAX_LEN)
sql_tokenizer.fit_on_texts(DATASET["Query"])  # Fit tokenizer on dataset

# Load the model using tf.saved_model.load and get the serving signature
loaded_model = tf.saved_model.load(MODEL_PATH)
model_predict = loaded_model.signatures["serving_default"]


def warm_up_model():
    """Sends a dummy request to the model to 'warm it up'."""
    dummy_query = "SELECT * FROM users WHERE id = 1"
    query_seq = sql_tokenizer.texts_to_sequences([dummy_query])
    input_tensor = tf.convert_to_tensor(query_seq, dtype=tf.float32)
    _ = model_predict(input_tensor)  # Make a dummy prediction to initialize the model
    print("Model warmed up and ready to serve requests.")


@app.route("/predict", methods=["POST"])
def predict():
    if not request.json or "query" not in request.json:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Tokenize and pad the input query using SQLTokenizer
        query = request.json["query"]
        query_seq = sql_tokenizer.texts_to_sequences([query])
        input_tensor = tf.convert_to_tensor(query_seq, dtype=tf.float32)

        # Use the loaded model's serving signature to make the prediction
        prediction = model_predict(input_tensor)

        # Check for valid output and extract the result
        if "output_0" not in prediction or prediction["output_0"].get_shape() != [1, 1]:
            return jsonify({"error": "Invalid model output"}), 500

        # Extract confidence and return the response
        return jsonify(
            {
                "confidence": float("%.4f" % prediction["output_0"].numpy()[0][0]),
            }
        )
    except Exception as e:
        # Log the error and return a proper error message
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    warm_up_model()
    app.run(host="0.0.0.0", port=8000, debug=True)
