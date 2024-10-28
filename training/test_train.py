import os
import pandas as pd
import pytest
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TFSMLayer
from sql_tokenizer import SQLTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer  # For old tokenizer

MAX_WORDS = 10000
MAX_LEN = 100
TOKENIZER_VOCAB_PATH = "sql_tokenizer_vocab.json"  # Path to saved vocabulary

MODELV1 = {
    "dataset": "dataset/sqli_dataset1.csv",
    "model_path": "sqli_model/1",
    "index": 0,
    "use_sql_tokenizer": False,
}
MODELV2 = {
    "dataset": "dataset/sqli_dataset2.csv",
    "model_path": "sqli_model/2",
    "index": 1,
    "use_sql_tokenizer": False,
}
MODELV3 = {
    "dataset": "dataset/sqli_dataset2.csv",
    "model_path": "sqli_model/3",
    "index": 2,
    "use_sql_tokenizer": True,
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

    # Select the appropriate tokenizer
    if request.param["use_sql_tokenizer"]:
        # Use SQLTokenizer for MODELV3
        tokenizer = SQLTokenizer(max_words=MAX_WORDS, max_len=MAX_LEN)

        # Load saved vocabulary if available
        if os.path.exists(TOKENIZER_VOCAB_PATH):
            tokenizer.load_token_index(TOKENIZER_VOCAB_PATH)
        else:
            tokenizer.fit_on_texts(data["Query"])
            tokenizer.save_token_index(
                TOKENIZER_VOCAB_PATH
            )  # Save for future consistency
    else:
        # Use the old Keras Tokenizer for MODELV1 and MODELV2
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
        ("select * from users where id=1 or 1=1;", [0.9202, 0.974, 0.3179]),
        ("select * from users where id='1' or 1=1--", [0.9202, 0.974, 0.3179]),
        ("select * from users", [0.00077, 0.0015, 0.0231]),
        ("select * from users where id=10000", [0.1483, 0.8893, 0.7307]),
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
    # Tokenize and pad the sample using the selected tokenizer
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

    # Check that prediction matches expected value within tolerance
    assert predicted_value == pytest.approx(sample[1][model["index"]], abs=0.05)
