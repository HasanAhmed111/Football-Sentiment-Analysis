import random

def predict_ensemble(text, model_name):

    # placeholder logic (replace with real models)
    sentiments = ["Positive", "Negative", "Neutral"]

    sentiment = random.choice(sentiments)
    confidence = random.uniform(0.70, 0.95)

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }