from tensorflow import keras

from src.data.loader import load_data
from src.data.preprocess import create_mapping, text_to_int

MODEL_PATH = 'models/sms-text-classifier.h5'
TRAIN_DIR = 'data/raw/train-data.tsv'

def predict(model_path, train_dir, pred_text):
    prediction, label = [], None

    df_train = load_data(train_dir)
    char2idx, _ = create_mapping(df_train)

    enoded_text = text_to_int(char2idx, pred_text)
    text_pad = keras.preprocessing.sequence.pad_sequences([enoded_text], 250)

    model = keras.models.load_model(model_path)
    probability = model.predict(text_pad)
    prediction.append(probability[0][0].item())

    if probability > 0.01:
        label = 'spam'
    else:
        label = 'ham'
    prediction.append(label)

    return (prediction)

if __name__ == "__main__":
    pred_text = input("Enter the text to predict: ")
    prediction = predict(MODEL_PATH, TRAIN_DIR, pred_text)
    print(f"Prediction: {prediction[0]}, Label: {prediction[1]}")





