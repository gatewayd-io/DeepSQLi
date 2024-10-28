import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def preprocess_text(data, max_words=10000, max_len=100):
    """Tokenize and pad text data."""
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data["Query"])
    sequences = tokenizer.texts_to_sequences(data["Query"])
    return pad_sequences(sequences, maxlen=max_len), tokenizer


def build_model(input_dim, output_dim=128):
    """Define and compile the CNN-BiLSTM model."""
    model = Sequential(
        [
            Embedding(input_dim=input_dim, output_dim=output_dim),
            Dropout(0.2),
            Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
            MaxPooling1D(pool_size=2),
            Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def calculate_f1_f2(precision, recall, beta=1):
    """Calculate F1 or F2 score based on precision and recall with given beta."""
    beta_squared = beta**2
    return (
        (1 + beta_squared)
        * (precision * recall)
        / (beta_squared * precision + recall + tf.keras.backend.epsilon())
    )


def plot_history(history):
    """Plot the training and validation loss, accuracy, precision, and recall."""
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(["loss", "accuracy", "precision", "recall"], start=1):
        plt.subplot(2, 2, i)
        plt.plot(history.history[metric], label=f"Training {metric.capitalize()}")
        plt.plot(
            history.history[f"val_{metric}"], label=f"Validation {metric.capitalize()}"
        )
        plt.title(metric.capitalize())
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")


# Main function
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <input_file> <output_dir>")
        sys.exit(1)

    # Load and preprocess data
    data = load_data(sys.argv[1])
    X, tokenizer = preprocess_text(data)
    y = data["Label"]

    # Initialize cross-validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "f2": []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"Training fold {fold}/{k_folds}")

        # Split the data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Build and train the model
        model = build_model(input_dim=len(tokenizer.word_index) + 1)
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1,
        )

        # Make predictions to manually calculate metrics
        y_val_pred = (model.predict(X_val) > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1_score = calculate_f1_f2(precision, recall, beta=1)
        f2_score = calculate_f1_f2(precision, recall, beta=2)

        # Collect fold metrics
        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["f1"].append(f1_score)
        fold_metrics["f2"].append(f2_score)

    # Calculate average metrics across folds
    avg_metrics = {metric: np.mean(scores) for metric, scores in fold_metrics.items()}
    print("\nCross-validation results:")
    for metric, value in avg_metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}")

    # Save the final model trained on the last fold
    model.export(sys.argv[2])
    plot_history(history)
