"""Deep Learning Model Training with LSTM

This Python script is used for training a deep learning model using
Long Short-Term Memory (LSTM) networks.

The script starts by importing necessary libraries. These include `sys`
for interacting with the system, `pandas` for data manipulation, `tensorflow`
for building and training the model, `sklearn` for splitting the dataset and
calculating metrics, and `numpy` for numerical operations.

The script expects two command-line arguments: the input file and the output directory.
If these are not provided, the script will exit with a usage message.

The input file is expected to be a CSV file, which is loaded into a pandas DataFrame.
The script assumes that this DataFrame has a column named "Query" containing the text
data to be processed, and a column named "Label" containing the target labels.

The text data is then tokenized using the `Tokenizer` class from
`tensorflow.keras.preprocessing.text` (TF/IDF). The tokenizer is fit on the text data
and then used to convert the text into sequences of integers. The sequences are then
padded to a maximum length of 100 using the `pad_sequences` function.

The data is split into a training set and a test set using the `train_test_split` function
from `sklearn.model_selection`. The split is stratified, meaning that the distribution of
labels in the training and test sets should be similar.

A Sequential model is created using the `Sequential` class from `tensorflow.keras.models`.
The model consists of an Embedding layer, an LSTM layer, and a Dense layer. The model is
compiled with the Adam optimizer and binary cross-entropy loss function, and it is trained
on the training data.

After training, the model is used to predict the labels of the test set. The predictions
are then compared with the true labels to calculate various performance metrics, including
accuracy, recall, precision, F1 score, specificity, and ROC. These metrics are printed to
the console.

Finally, the trained model is saved in the SavedModel format to the output directory
specified by the second command-line argument.
"""

import sys
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.metrics import Accuracy, Recall, Precision, F1Score
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
model.add(Embedding(MAX_WORDS, 128))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        Accuracy(),
        Recall(),
        Precision(),
        F1Score(),
    ],
)

# Train model
model.fit(X_train, y_train, epochs=11, batch_size=32)

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
model.export(sys.argv[2])
