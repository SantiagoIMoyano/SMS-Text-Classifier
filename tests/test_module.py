import pytest
from src.model.inference import predict

MODEL_PATH = 'models/sms-text-classifier.h5'
TRAIN_DIR  = 'data/raw/train-data.tsv'

@pytest.mark.parametrize("message, expected_label", [
    ("how are you doing today", "ham"),
    ("sale today! to stop texts call 98912460324", "spam"),
    ("i dont want to go. can we try it a different day? available sat", "ham"),
    ("our new mobile video service is live. just install on your phone to start watching.", "spam"),
    ("you have won £1000 cash! call to claim your prize.", "spam"),
    ("i'll bring it tomorrow. don't forget the milk.", "ham"),
    ("wow, is your arm alright. that happened to me one time too", "ham"),
])
def test_predict_labels(message, expected_label):
    pred_label = predict(MODEL_PATH, TRAIN_DIR, message)
    assert pred_label[1] == expected_label