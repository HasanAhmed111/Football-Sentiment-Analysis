def explain_sentiment(text):

    text = text.lower()

    positive_keywords = ["win", "great", "amazing", "goal", "best", "win"]
    negative_keywords = ["bad", "worst", "referee", "injury", "fail"]

    signals = []

    for word in positive_keywords:
        if word in text:
            signals.append(f"Positive signal detected: {word}")

    for word in negative_keywords:
        if word in text:
            signals.append(f"Negative signal detected: {word}")

    if len(signals) == 0:
        signals.append("No strong emotional signals detected")

    sentiment = "Neutral"

    if len([s for s in signals if "Positive" in s]) > len([s for s in signals if "Negative" in s]):
        sentiment = "Positive"
    elif len([s for s in signals if "Negative" in s]) > len([s for s in signals if "Positive" in s]):
        sentiment = "Negative"

    return {
        "summary": f"Detected {sentiment} sentiment based on keyword context",
        "signals": signals
    }