import pandas as pd
import pytest
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TFSMLayer
import numpy as np


MAX_WORDS = 10000
MAX_LEN = 100
MODELV1 = {
    "dataset": "dataset/sqli_dataset1.csv",
    "model_path": "sqli_model/1",
    "index": 0,
}
MODELV2 = {
    "dataset": "dataset/sqli_dataset2.csv",
    "model_path": "sqli_model/2",
    "index": 1,
}


@pytest.fixture(
    params=[
        MODELV1,
        MODELV2,
    ],
)
def model(request):
    # Load dataset
    data = pd.read_csv(request.param["dataset"])

    # Load TF model from SavedModel
    sqli_model = TFSMLayer(request.param["model_path"], call_endpoint="serving_default")

    # Tokenize the sample
    tokenizer = Tokenizer(num_words=MAX_WORDS, filters="")
    tokenizer.fit_on_texts(data["Query"])

    return {
        "tokenizer": tokenizer,
        "sqli_model": sqli_model,
        "index": request.param["index"],
    }


@pytest.mark.parametrize(
    "sample",
    [
        ("select * from users where id=1 or 1=1;", [99.99, 99.83]),
        ("select * from users where id='1' or 1=1--", [92.02, 99.83]),
        ("select * from users", [0.077, 0.08]),
        ("select * from users where id=10000", [14.83, 97.32]),
        ("select '1' union select 'a'; -- -'", [99.99, 97.32]),
        (
            "select '' union select 'malicious php code' \g /var/www/test.php; -- -';",
            [99.99, 99.99],
        ),
        (
            "select '' || pg_sleep((ascii((select 'a' limit 1)) - 32) / 2); -- -';",
            [99.99, 99.99],
        ),
    ],
)
def test_sqli_model(model, sample):
    # Vectorize the sample
    sample_seq = model["tokenizer"].texts_to_sequences([sample[0]])
    sample_vec = pad_sequences(sample_seq, maxlen=MAX_LEN)

    # Predict sample
    predictions = model["sqli_model"](sample_vec)

    # Scale up to 100
    print(predictions["dense"].numpy() * 100)  # Debugging purposes (prints on error)
    assert predictions["dense"].numpy() * 100 == pytest.approx(
        np.array([[sample[1][model["index"]]]]), 0.1
    )
