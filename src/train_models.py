import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

# -----------------------------------
# BASE DIRECTORY
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------
# LOAD DATASET
# -----------------------------------

dataset_path = os.path.join(
    BASE_DIR,
    "..",
    "dataset",
    "processed",
    "cleaned_data.csv"
)

df = pd.read_csv(dataset_path, encoding="latin1")

# -----------------------------------
# FEATURES + LABELS
# -----------------------------------

X_text = df["clean_text"].astype(str)

# LABEL ENCODING
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["Sentiment"]
)

# -----------------------------------
# IMPROVED TF-IDF
# -----------------------------------

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True,
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(X_text)

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------
# MODEL DICTIONARY
# -----------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight='balanced'
    ),

    "Naive Bayes": MultinomialNB(),

    "SVM": LinearSVC(
        C=1.5,
        class_weight='balanced'
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=20,
        class_weight='balanced'
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        n_jobs=-1
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss'
    )
}

# -----------------------------------
# RESULTS STORAGE
# -----------------------------------

results = []

# -----------------------------------
# REPORT DIRECTORIES
# -----------------------------------

reports_dir = os.path.join(
    BASE_DIR,
    "..",
    "reports"
)

figures_dir = os.path.join(
    reports_dir,
    "figures"
)

cm_dir = os.path.join(
    reports_dir,
    "confusion_matrices"
)

metrics_dir = os.path.join(
    reports_dir,
    "metrics"
)

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(cm_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# -----------------------------------
# TRAIN ALL MODELS
# -----------------------------------

for name, model in models.items():

    print("\n")
    print("=" * 60)

    print(f"TRAINING: {name}")

    print("=" * 60)

    # TRAIN
    model.fit(X_train, y_train)

    # PREDICT
    predictions = model.predict(X_test)

    # METRICS
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average='macro'
    )

    recall = recall_score(
        y_test,
        predictions,
        average='macro'
    )

    f1 = f1_score(
        y_test,
        predictions,
        average='macro'
    )

    # STORE RESULTS
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # -----------------------------------
    # PRINT REPORT
    # -----------------------------------

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_
        )
    )

    # -----------------------------------
    # CONFUSION MATRIX
    # -----------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.title(f"{name} Confusion Matrix")

    plt.colorbar()

    plt.xticks(
        ticks=range(len(label_encoder.classes_)),
        labels=label_encoder.classes_,
        rotation=45
    )

    plt.yticks(
        ticks=range(len(label_encoder.classes_)),
        labels=label_encoder.classes_
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    cm_path = os.path.join(
        cm_dir,
        f"{name}_cm.png"
    )

    plt.tight_layout()

    plt.savefig(cm_path)

    plt.close()

    # -----------------------------------
    # SAVE MODEL
    # -----------------------------------

    model_name = name.lower().replace(" ", "_")

    model_path = os.path.join(
        BASE_DIR,
        "..",
        "models",
        f"{model_name}.pkl"
    )

    joblib.dump(model, model_path)

# -----------------------------------
# SAVE TF-IDF VECTORIZER
# -----------------------------------

vectorizer_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "tfidf_vectorizer.pkl"
)

joblib.dump(vectorizer, vectorizer_path)

# -----------------------------------
# SAVE LABEL ENCODER
# -----------------------------------

encoder_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "label_encoder.pkl"
)

joblib.dump(label_encoder, encoder_path)

# -----------------------------------
# RESULTS DATAFRAME
# -----------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

# -----------------------------------
# PRINT RESULTS
# -----------------------------------

print("\n")
print("=" * 60)

print("FINAL MODEL COMPARISON")

print("=" * 60)

print(results_df)

# -----------------------------------
# SAVE RESULTS CSV
# -----------------------------------

results_csv_path = os.path.join(
    metrics_dir,
    "model_comparison.csv"
)

results_df.to_csv(
    results_csv_path,
    index=False
)

# -----------------------------------
# ACCURACY GRAPH
# -----------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.xticks(rotation=45)

plt.title("Model Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.tight_layout()

graph_path = os.path.join(
    figures_dir,
    "accuracy_comparison.png"
)

plt.savefig(graph_path)

plt.close()

print("\n")
print("ALL MODELS TRAINED SUCCESSFULLY")