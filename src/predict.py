import os
import joblib

from preprocessing import clean_text

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
model_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "logistic_model.pkl"
)

vectorizer_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "tfidf_vectorizer.pkl"
)

# Load model and vectorizer
model = joblib.load(model_path)

vectorizer = joblib.load(vectorizer_path)

def predict_sentiment(text):

    cleaned = clean_text(text)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)

    return prediction[0]


if __name__ == "__main__":

    user_text = input("Enter Sports Comment: ")

    result = predict_sentiment(user_text)

    print(f"Predicted Sentiment: {result}")