import os
from tensorflow.keras.models import load_model

from src.data.loader import download_data, load_data
from src.data.preprocess import preprocess_data

URL = 'https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv'
DEST = 'data/raw'
FILE = 'valid-data.tsv'
MODEL_PATH = 'models/sms-text-classifier.h5'
TRAIN_DIR = 'data/raw/train-data.tsv'

def evaluate_model(data_url, dest_dir, file, model_path, train_path):
    download_data(data_url, dest_dir)
    file_path = os.path.join(dest_dir, file)

    df_train = load_data(train_path)
    df_test = load_data(file_path)

    data_pad, labels = preprocess_data(df_train, df_test, train=False)

    model = load_model(model_path)
    loss, acc = model.evaluate(data_pad, labels)

    print(f"Model Evaluation:\nLoss: {loss}\Accuracy: {acc}")

if __name__ == "__main__":
    evaluate_model(URL, DEST, FILE, MODEL_PATH, TRAIN_DIR)