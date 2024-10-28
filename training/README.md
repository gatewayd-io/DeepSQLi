# SQL Injection Detection Model Training and Tokenization

This repository contains code for training SQL injection detection models using various deep learning architectures and tokenization methods. The models are saved in the [`sqli_model`](../sqli_model/) directory, organized by versions based on their architecture and tokenization strategy.

## Overview of Training Files

### Model Training Scripts

- **[`train.py`](train.py)**: This script trains models saved in [`sqli_model/1`](../sqli_model/1/) and [`sqli_model/2`](../sqli_model/2/). It utilizes an **LSTM-based architecture** and the default Keras tokenizer.
- **[`train_v3.py`](train_v3.py)**: This script trains the model saved in [`sqli_model/3`](../sqli_model/3/), which uses a **Deep Learning CNN-LSTM hybrid model** with a custom SQL tokenizer designed to handle SQL syntax and injection patterns more effectively.

### Tokenization Methods

Each training script employs a different tokenization strategy, suitable for the specific model architecture:

- **Keras Default Tokenizer (`train.py` for `sqli_model/1` and `sqli_model/2`)**:
  - The default Keras tokenizer (`tensorflow.keras.preprocessing.text.Tokenizer`) is used in `train.py`. It performs basic tokenization by splitting the text into words and mapping each word to an integer. This simple approach is effective for general text but may miss some nuances specific to SQL syntax.

- **Custom SQL Tokenizer (`train_v3.py` for `sqli_model/3`)**:
  - The `train_v3.py` script employs a [custom SQL tokenizer](sql_tokenizer.py) specifically designed to handle SQL syntax and keywords. This tokenizer tokenizes SQL queries by recognizing SQL-specific patterns, operators, and punctuation, providing a more robust representation for SQL injection detection. This method captures complex SQL expressions that are crucial for detecting injection patterns accurately.

## Model Architectures

- **LSTM Model** (`train.py` for `sqli_model/1` and `sqli_model/2`):
  - The models in `sqli_model/1` and `sqli_model/2` are trained using an LSTM (Long Short-Term Memory) network. LSTMs are particularly suited for sequential data, making them effective in capturing dependencies in SQL query patterns.

- **CNN-LSTM Hybrid Model** (`train_v3.py` for sqli_model/3):
  - The model in `sqli_model/3` is a hybrid CNN-LSTM model. It combines convolutional layers to detect local patterns in SQL syntax and LSTM layers to capture sequential dependencies. This architecture, combined with the custom SQL tokenizer, enhances the model’s ability to detect complex injection patterns.

## How to Use

1. **Training**:
   - Run `train.py` to train models using the default Keras tokenizer and save them in `sqli_model/1` or `sqli_model/2`.
   - Run `train_v3.py` to train the CNN-LSTM model with the custom SQL tokenizer and save it in `sqli_model/3`.

2. **Tokenization**:
   - The default tokenizer from Keras is used automatically within `train.py`.
   - For `train_v3.py`, the custom SQL tokenizer defined in [sql_tokenizer.py](sql_tokenizer.py) is automatically used.

## File Structure

- **[`train.py`](train.py)** - Trains models with default Keras tokenizer and LSTM architecture.
- **[`train_v3.py`](train_v3.py)** - Trains model with custom SQL tokenizer and CNN-LSTM architecture.
- **[`sql_tokenizer.py`](sql_tokenizer.py)** - Custom SQL tokenizer for handling SQL-specific patterns, used by `train_v3.py`.
