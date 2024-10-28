import pandas as pd
import pytest
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TFSMLayer


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
MODELV3 = {
    "dataset": "dataset/sqli_dataset2.csv",
    "model_path": "sqli_model/3",
    "index": 2,
}


@pytest.fixture(
    params=[
        MODELV1,
        MODELV2,
        MODELV3,
    ],
)
def model(request):
    # Load dataset
    prefix = ""
    data = None
    try:
        data = pd.read_csv(request.param["dataset"])
    except FileNotFoundError:
        # Check if the dataset is in the parent directory
        prefix = "../"
        data = pd.read_csv(prefix + request.param["dataset"])

    # Load TF model using TFSMLayer with the serving_default endpoint
    model_path = prefix + request.param["model_path"]
    sqli_model = TFSMLayer(model_path, call_endpoint="serving_default")

    # Tokenizer setup
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
        ("select * from users where id=1 or 1=1;", [0.9202, 0.974, 0.0022]),
        ("select * from users where id='1' or 1=1--", [0.9202, 0.974, 0.0022]),
        ("select * from users", [0.00077, 0.0015, 0.0231]),
        ("select * from users where id=10000", [0.1483, 0.8893, 0.0008]),
        ("select '1' union select 'a'; -- -'", [0.9999, 0.9732, 0.0139]),
        (
            "select '' union select 'malicious php code' \\g /var/www/test.php; -- -';",
            [0.9999, 0.8065, 0.0424],
        ),
        (
            "select '' || pg_sleep((ascii((select 'a' limit 1)) - 32) / 2); -- -';",
            [0.9999, 0.9999, 0.01543],
        ),
    ],
)
def test_sqli_model(model, sample):
    # Vectorize the sample
    sample_seq = model["tokenizer"].texts_to_sequences([sample[0]])
    sample_vec = pad_sequences(sample_seq, maxlen=MAX_LEN)

    # Predict sample
    predictions = model["sqli_model"](sample_vec)

    # Extract the prediction result
    output_key = "output_0" if "output_0" in predictions else "dense"
    predicted_value = predictions[output_key].numpy()[0][0]

    print(
        f"Predicted: {predicted_value:.4f}, Expected: {sample[1][model['index']]:.4f}"
    )

    assert predicted_value == pytest.approx(sample[1][model["index"]], abs=0.05)
