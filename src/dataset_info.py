import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(
    BASE_DIR,
    "..",
    "dataset",
    "processed",
    "cleaned_data.csv"
)

df = pd.read_csv(input_path , encoding="latin1")
print(df.shape, df["Sentiment"].value_counts())
print(df['Source of Tweet'].nunique())

import os
import matplotlib.pyplot as plt

# -----------------------------------
# CREATE PLOTS FOLDER AUTOMATICALLY
# -----------------------------------

plots_dir = "plots"

os.makedirs(
    plots_dir,
    exist_ok=True
)

# -----------------------------------
# SENTIMENT COUNTS
# -----------------------------------

sentiment_counts = df["Sentiment"].value_counts()
print(sentiment_counts)

# -----------------------------------
# BAR GRAPH
# -----------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    sentiment_counts.index,
    sentiment_counts.values
)

plt.title("Sentiment Category Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Number of Tweets")

# VALUES ON TOP
for i, value in enumerate(sentiment_counts.values):
    plt.text(
        i,
        value + 50,
        str(value),
        ha='center'
    )

# -----------------------------------
# SAVE GRAPH
# -----------------------------------

save_path = os.path.join(
    plots_dir,
    "sentiment_distribution_bar.png"
)

plt.savefig(save_path)

plt.show()

print(f"Plot saved at: {save_path}")