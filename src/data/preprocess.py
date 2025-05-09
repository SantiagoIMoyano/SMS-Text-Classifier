import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_mapping(df_train):
    corpus = " ".join(df_train["text"].tolist())
    vocab = sorted(set(corpus))

    # Crear mapeo: de carácter a índice y viceversa
    char2idx = {u: i for i, u in enumerate(vocab)}

    return char2idx, len(vocab)

def text_to_int(char2idx, text):
    return np.array([char2idx[c] for c in text if c in char2idx])

def preprocess_data(df_train, df_test=None, train=True):
    max_len = 250

    char2idx, vocab_size = create_mapping(df_train)

    if train:
        df_train["int_sequence"] = df_train["text"].apply(lambda txt: text_to_int(char2idx, txt))
        train_sequences = df_train["int_sequence"].tolist()
        train_labels = df_train["label"].values
        labels = np.where(train_labels == "spam", 1, 0)
        data_pad = pad_sequences(train_sequences, max_len)

        return data_pad, labels, vocab_size
    else:
        df_test["int_sequence"] = df_test["text"].apply(lambda txt: text_to_int(char2idx, txt))
        test_sequences = df_test["int_sequence"].tolist()
        test_labels = df_test["label"].values
        labels = np.where(test_labels == "spam", 1, 0)
        data_pad = pad_sequences(test_sequences, max_len)
        
        return data_pad, labels
