import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------------
# BASE DIRECTORY
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------
# CREATE PLOTS DIRECTORY
# -----------------------------------

plots_dir = os.path.join(
    BASE_DIR,
    "..",
    "plots"
)

os.makedirs(plots_dir, exist_ok=True)

# -----------------------------------
# YOUR ACTUAL MODEL RESULTS
# -----------------------------------

results = {
    "Model": [
        "Logistic Regression",
        "Naive Bayes",
        "SVM",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "XGBoost",
        "LSTM",
        "BiLSTM",
        "DistilBERT"
    ],

    "Accuracy": [
        0.708102,
        0.693674,
        0.692120,
        0.510322,
        0.633518,
        0.496559,
        0.676804,
        0.38,  
        0.68,  
        0.78   
    ]
}

# -----------------------------------
# CREATE DATAFRAME
# -----------------------------------

df = pd.DataFrame(results)

# SORT BY ACCURACY
df = df.sort_values(
    by="Accuracy",
    ascending=False
)

# -----------------------------------
# CREATE GRAPH
# -----------------------------------

plt.figure(figsize=(14, 7))

bars = plt.bar(
    df["Model"],
    df["Accuracy"]
)

# TITLE + LABELS
plt.title(
    "Sports Sentiment Analysis Model Comparison",
    fontsize=18
)

plt.xlabel(
    "Models",
    fontsize=14
)

plt.ylabel(
    "Accuracy",
    fontsize=14
)

plt.xticks(
    rotation=45,
    fontsize=11
)

# -----------------------------------
# VALUES ON TOP OF BARS
# -----------------------------------

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.005,
        f"{height:.2f}",
        ha='center',
        fontsize=10
    )

# -----------------------------------
# SAVE GRAPH
# -----------------------------------

plot_path = os.path.join(
    plots_dir,
    "model_comparison_graph.png"
)

plt.tight_layout()

plt.savefig(
    plot_path,
    dpi=300
)

plt.show()

print("\nGraph Saved Successfully")
print("Saved at:", plot_path)