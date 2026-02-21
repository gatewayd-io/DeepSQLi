# sql_tokenizer.py
import re
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SQLTokenizer:
    def __init__(self, max_words=10000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.token_index = {}

    def tokenize(self, query):
        # Define a regex pattern for SQL tokens (operators, punctuation, keywords)
        pattern = r"[\w']+|[=><!]+|--|/\*|\*/|;|\(|\)|,|\*|\||\s+"
        tokens = re.findall(pattern, query.lower())
        return tokens

    def fit_on_texts(self, queries):
        # Build a token index based on the provided queries
        all_tokens = set()
        for query in queries:
            tokens = self.tokenize(query)
            all_tokens.update(tokens)
        # Sort for deterministic ordering, then limit to max_words
        all_tokens = sorted(all_tokens)[: self.max_words]
        self.token_index = {token: i + 1 for i, token in enumerate(all_tokens)}

    def texts_to_sequences(self, queries):
        # Convert queries to sequences of token IDs
        sequences = []
        for query in queries:
            tokens = self.tokenize(query)
            sequence = [self.token_index.get(token, 0) for token in tokens]
            sequences.append(sequence)
        return pad_sequences(sequences, maxlen=self.max_len)

    def save_token_index(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.token_index, f)

    def load_token_index(self, filepath):
        with open(filepath, "r") as f:
            self.token_index = json.load(f)
