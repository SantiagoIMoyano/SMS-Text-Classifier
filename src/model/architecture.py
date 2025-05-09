from tensorflow import keras

def build_model(vocab_size):
    model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 64),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.000778)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])

    return model