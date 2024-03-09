import pandas as pd
import pytest
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np


MAX_WORDS = 10000
MAX_LEN = 100


@pytest.fixture
def model():

    # Load dataset
    data = pd.read_csv("dataset/sqli_dataset.csv")

    # Load TF model from SavedModel
    sqli_model = load_model("sqli_model/1")

    # Tokenize the sample
    tokenizer = Tokenizer(num_words=MAX_WORDS, filters="")
    tokenizer.fit_on_texts(data["Query"])

    return {"tokenizer": tokenizer, "sqli_model": sqli_model}


@pytest.mark.parametrize("sample",
    [
        ("select * from users where id='1' or 1=1--", [92.02]),
        ("select * from users", [0.077]),
        ("select * from users where id=10000", [14.83]),
        ("select '1' union select 'a'; -- -'", [99.99]),
        ("select '' union select 'malicious php code' \g /var/www/test.php; -- -';", [99.99]),
        ("select '' || pg_sleep((ascii((select 'a' limit 1)) - 32) / 2); -- -';", [99.99])
    ]
)
def test_sqli_model(model, sample):
    # Vectorize the sample
    sample_seq = model["tokenizer"].texts_to_sequences([sample[0]])
    sample_vec = pad_sequences(sample_seq, maxlen=MAX_LEN)

    # Predict sample
    predictions = model["sqli_model"].predict(sample_vec)

    # Scale up to 100
    assert predictions * 100 == pytest.approx(
        np.array([sample[1]]), 0.1
    )
