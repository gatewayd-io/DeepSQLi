import pandas as pd
import pytest
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np


def test_sqli_model():
    MAX_WORDS = 10000
    MAX_LEN = 100

    # Load dataset
    data = pd.read_csv("dataset/sqli_dataset.csv")

    # Load TF model from SavedModel
    sqli_model = load_model("sqli_model")

    # Create a sample SQL injection data
    sample = [
        "select * from users where id='1' or 1=1--",
        "select * from users",
        "select * from users where id=10000",
        (
            "select * from test where id=1 UNION ALL "
            "SELECT NULL FROM INFORMATION_SCHEMA.COLUMNS WHERE 1=0; --;"
        ),
    ]

    # Tokenize the sample
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(data["Query"])

    # Vectorize the sample
    sample_seq = tokenizer.texts_to_sequences(sample)
    sample_vec = pad_sequences(sample_seq, maxlen=MAX_LEN)

    # Predict sample
    predictions = sqli_model.predict(sample_vec)
    # Scale up to 100
    assert predictions * 100 == pytest.approx(
        np.array([[99.99], [0.005], [0.055], [99.99]]), 0.1
    )
