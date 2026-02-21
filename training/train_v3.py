import sys
import os
import pandas as pd
import tensorflow as tf
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
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from sql_tokenizer import SQLTokenizer


def load_data(file_path):
    """Load data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def preprocess_text(data, max_words=10000, max_len=100):
    """Tokenize and pad text data using SQLTokenizer."""
    tokenizer = SQLTokenizer(max_words=max_words)
    tokenizer.fit_on_texts(data["Query"])
    sequences = tokenizer.texts_to_sequences(data["Query"])
    return sequences, tokenizer


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
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
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
    available_metrics = history.history.keys()  # Check which metrics are available
    plt.figure(figsize=(12, 8))

    # Define metrics to plot
    metrics_to_plot = ["loss", "accuracy", "precision", "recall"]
    for i, metric in enumerate(metrics_to_plot, start=1):
        if metric in available_metrics:
            plt.subplot(2, 2, i)
            plt.plot(history.history[metric], label=f"Training {metric.capitalize()}")
            plt.plot(
                history.history[f"val_{metric}"],
                label=f"Validation {metric.capitalize()}",
            )
            plt.title(metric.capitalize())
            plt.xlabel("Epochs")
            plt.ylabel(metric.capitalize())
            plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_v3.py <input_file> <output_dir>")
        sys.exit(1)

    MAX_WORDS = 10000
    MAX_LEN = 100
    EPOCHS = 50
    BATCH_SIZE = 32

    # Load and preprocess data
    data = load_data(sys.argv[1])
    X, tokenizer = preprocess_text(data, max_words=MAX_WORDS)
    y = data["Label"].values  # Convert to NumPy array for compatibility with KFold

    # Save the deterministic vocabulary for inference
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(script_dir, "sql_tokenizer_vocab.json")
    tokenizer.save_token_index(vocab_path)
    print(f"Saved tokenizer vocabulary ({len(tokenizer.token_index)} tokens) to {vocab_path}")

    # Initialize cross-validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "f2": []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"Training fold {fold}/{k_folds}")

        # Split the data
        X_train, X_val = np.array(X)[train_idx], np.array(X)[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        # Build and train the model
        model = build_model(input_dim=len(tokenizer.token_index) + 1)
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        history = model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,
            callbacks=[early_stopping],
            verbose=1,
        )

        # Make predictions to calculate metrics
        y_val_pred = (model.predict(X_val) > 0.8).astype(int)
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = calculate_f1_f2(precision, recall, beta=1)
        f2 = calculate_f1_f2(precision, recall, beta=2)

        # Collect fold metrics
        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["f1"].append(f1)
        fold_metrics["f2"].append(f2)

    # Calculate and display average metrics across folds
    avg_metrics = {metric: np.mean(scores) for metric, scores in fold_metrics.items()}
    print("\nCross-validation results:")
    for metric, value in avg_metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}")

    # Save the final model trained on the last fold
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.export(output_dir)

    # Plot training history of the last fold
    plot_history(history)
