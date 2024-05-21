import sys
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Embedding,
    Flatten,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.metrics import Accuracy, Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
import numpy as np
import matplotlib.pyplot as plt


# Check if the input file and output directory are provided
if len(sys.argv) != 3:
    print("Usage: python train.py <input_file> <output_dir>")
    sys.exit(1)

# Load dataset
data = pd.read_csv(sys.argv[1])

# Define parameters
MAX_WORDS = 10000
MAX_LEN = 100

# Use Tokenizer to encode text
tokenizer = Tokenizer(num_words=MAX_WORDS, filters="")
tokenizer.fit_on_texts(data["Query"])
sequences = tokenizer.texts_to_sequences(data["Query"])

# Pad the text sequence
X = pad_sequences(sequences, maxlen=MAX_LEN)

# Split the training set and test set
y = data["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create CNN-BiLSTM model
model = Sequential()
model.add(Embedding(MAX_WORDS, 128))
model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        Accuracy(),
        Recall(),
        Precision(),
    ],
)

# Define early stopping callback with a rollback of 5
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Train model with early stopping
history = model.fit(
    X_train,
    y_train,
    epochs=50,  # Maximum number of epochs
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1,
)

# Predict test set
y_pred = model.predict(X_test, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate model performance indicators
accuracy = accuracy_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes, zero_division=1)
precision = precision_score(y_test, y_pred_classes, zero_division=1)
f1 = f1_score(y_test, y_pred_classes, zero_division=1)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_classes).ravel()

# Output performance indicators
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("F1-score: {:.2f}%".format(f1 * 100))
print("Specificity: {:.2f}%".format(tn / (tn + fp) * 100))
print("ROC: {:.2f}%".format(tp / (tp + fn) * 100))

# Save model as SavedModel format
model.export(sys.argv[2])


# Plot the training history
def plot_history(history):
    plt.figure(figsize=(12, 8))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot precision
    plt.subplot(2, 2, 3)
    plt.plot(history.history["precision"], label="Training Precision")
    plt.plot(history.history["val_precision"], label="Validation Precision")
    plt.title("Precision")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.legend()

    # Plot recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history["recall"], label="Training Recall")
    plt.plot(history.history["val_recall"], label="Validation Recall")
    plt.title("Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")


plot_history(history)
