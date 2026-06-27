import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------------
# INITIALIZE NLTK & REQUISITES
# -----------------------------------

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------------
# TEXT CLEANING FUNCTION
# -----------------------------------

def clean_text(text):
    text = str(text)

    # LOWERCASE
    text = text.lower()

    # REMOVE URLS
    text = re.sub(r"http\S+", "", text)

    # REMOVE MENTIONS
    text = re.sub(r"@\w+", "", text)

    # REMOVE HASHTAG SYMBOL
    text = re.sub(r"#", "", text)

    # REMOVE NUMBERS
    text = re.sub(r"\d+", "", text)

    # REMOVE SPECIAL CHARACTERS
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # TOKENIZATION
    words = text.split()

    # REMOVE STOPWORDS + LEMMATIZATION
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------------
# BASE DIR
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------
# LOAD RAW DATASET
# -----------------------------------

input_path = os.path.join(
    BASE_DIR,
    "..",
    "dataset",
    "raw",
    "fifa_world_cup_2022_tweets.csv"
)

df = pd.read_csv(input_path)

# -----------------------------------
# CLEAN TEXT
# -----------------------------------

df["clean_text"] = df["Tweet"].apply(clean_text)

# -----------------------------------
# REMOVE EMPTY ROWS
# -----------------------------------

df = df[df["clean_text"].str.strip() != ""]

# -----------------------------------
# SAVE CLEANED DATASET
# -----------------------------------

output_path = os.path.join(
    BASE_DIR,
    "..",
    "dataset",
    "processed",
    "cleaned_data.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("Preprocessing Completed Successfully")