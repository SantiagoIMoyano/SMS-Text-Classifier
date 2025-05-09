import os

from src.model.architecture import build_model
from src.data.loader import download_data, load_data
from src.data.preprocess import preprocess_data

URL = 'https://cdn.freecodecamp.org/project-data/sms/train-data.tsv'
DEST = 'data/raw'
FILE = 'train-data.tsv'

def train_model(data_url, dest_dir, file):
    download_data(data_url, dest_dir)
    file_path = os.path.join(dest_dir, file)

    df_train = load_data(file_path)
    data_pad, labels, vocab_size = preprocess_data(df_train, train=True)

    model = build_model(vocab_size)
    model.fit(data_pad, labels, epochs=10, shuffle=True)

    os.makedirs('models', exist_ok=True)
    model_path = 'models/sms-text-classifier.h5'
    model.save(model_path)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model(URL, DEST, FILE)

