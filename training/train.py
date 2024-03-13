# Deep Learning Model Training with LSTM

import sys
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
import numpy as np

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

# Create LSTM model
model = Sequential()
model.add(Embedding(MAX_WORDS, 128, input_length=MAX_LEN))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict test set
y_pred = model.predict(X_test)
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
save_model(model, sys.argv[2])
