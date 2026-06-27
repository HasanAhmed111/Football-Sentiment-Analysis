# ⚽ Football Sentiment Analysis & Intelligent Match Query System

A comprehensive **Natural Language Processing (NLP)** project that performs sentiment analysis on football-related text using **Traditional Machine Learning, Deep Learning, and Transformer-based models**. The project also includes an intelligent football query system capable of retrieving match-related information from a football discussion corpus through an interactive **Streamlit dashboard**.

---

# 📌 Project Overview

Football generates millions of discussions daily across Twitter, Reddit, sports forums, and news platforms. Understanding public opinion manually is nearly impossible due to the huge volume of textual data.

This project automatically analyzes football-related text to determine its sentiment while also allowing users to ask football-related questions such as match results, team comparisons, goals, cards, competitions, and player performance.

The project combines:

- Natural Language Processing (NLP)
- Machine Learning
- Deep Learning
- Transformer Models
- Rule-Based NLP
- Interactive Data Visualization
- Intelligent Query Processing

into one complete application.

---

# ✨ Features

## 🔹 Sentiment Analysis

- Single Text Sentiment Analysis
- Batch Sentiment Analysis
- Rule-Based Sentiment Analysis
- Machine Learning Prediction
- Deep Learning Prediction
- Transformer-Based Prediction
- Subjectivity Analysis
- Confidence Scores
- Football Phrase Detection

---

## 🔹 Football Query Engine

Supports football-related queries including:

- Team vs Team
- Match Result
- Goal Scorers
- Yellow Cards
- Red Cards
- Match Statistics
- Competitions
- Match Discussions
- Team Information

---

## 🔹 Interactive Dashboard

Built using **Streamlit**

Features include:

- Modern Responsive Blue UI
- Light/Dark Theme
- Model Comparison
- Batch Processing
- Query System
- Match Statistics
- Sentiment Visualization
- Performance Metrics

---

# 🧠 Machine Learning Models

The project trains and compares multiple machine learning algorithms.

| Model | Purpose |
|--------|----------|
| Logistic Regression | Baseline linear classifier |
| Naive Bayes | Probabilistic text classifier |
| Support Vector Machine (SVM) | Maximum-margin classifier |
| Decision Tree | Rule-based classifier |
| Random Forest | Ensemble decision trees |
| K-Nearest Neighbors (KNN) | Distance-based classifier |
| XGBoost | Gradient Boosting classifier |

---

# 🤖 Deep Learning Models

The project also implements neural network-based models.

- LSTM
- Bidirectional LSTM (BiLSTM)

These models learn contextual relationships within football-related text and improve sequential understanding compared to traditional ML models.

---

# 🚀 Transformer Model

The project includes a fine-tuned Transformer model.

- DistilBERT

DistilBERT provides contextual word embeddings and captures semantic relationships within football discussions more effectively than classical approaches.

---

# 📊 Model Performance

## Traditional Machine Learning

| Model | Accuracy |
|--------|-----------|
| Logistic Regression | **70.81%** |
| Naive Bayes | **69.37%** |
| SVM | **69.21%** |
| XGBoost | **67.68%** |
| Random Forest | **63.35%** |
| Decision Tree | **51.03%** |
| KNN | **49.65%** |

---

## Deep Learning

| Model | Accuracy |
|--------|-----------|
| BiLSTM | **66.95%** |
| LSTM | **38%** |

---

## Transformer

| Model | Accuracy |
|--------|-----------|
| DistilBERT | *(Update with final accuracy after training)* |

---

# 📂 Dataset

The project uses football-related tweets and discussions collected from the FIFA World Cup dataset.

Dataset preprocessing includes:

- Lowercase conversion
- URL removal
- Mention removal
- Emoji removal
- Punctuation removal
- Stopword removal
- Lemmatization
- Text normalization
- Duplicate removal

---

# 🏗 Project Structure

```
Football-Sentiment-Analysis/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── xgboost.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── lstm_model.h5
│   ├── bilstm_model.h5
│   └── distilbert_model/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_training.ipynb
│   ├── ML_PROJECT_DEEPLEARNING.ipynb
│   └── TRANSFORMER_TRAINING.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   ├── train_models.py
│   ├── dataset_info.py
│   └── model_comparison_graph.py
│
├── utils/
│   ├── ensemble.py
│   └── sentiment_explainer.py
│
└── plots/
```

---

# 🛠 Technologies Used

## Programming Language

- Python

---

## Libraries

- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Transformers
- PyTorch
- HuggingFace Datasets
- Streamlit
- Matplotlib
- Pickle

---

## NLP Techniques

- TF-IDF Vectorization
- Tokenization
- Padding
- Lemmatization
- Text Cleaning
- Phrase Detection
- Rule-Based Sentiment Analysis

---

# 📈 Data Science Workflow

The project follows a complete Data Science pipeline:

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Text Preprocessing
5. Feature Engineering (TF-IDF)
6. Machine Learning Training
7. Deep Learning Training
8. Transformer Fine-Tuning
9. Model Evaluation
10. Streamlit Deployment

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Football-Sentiment-Analysis.git

cd Football-Sentiment-Analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 💬 Example Queries

Football Query System

```
Manchester United vs Liverpool

Who won Argentina vs France?

Liverpool vs Manchester City

Did Messi score?

Show goals in Argentina vs France

Show yellow cards in Brazil vs Croatia

What competition was this match?

Show match statistics
```

Sentiment Analysis

```
What an unbelievable goal by Messi!

The referee completely ruined the match.

Manchester United played well today.

That was the worst defending I have ever seen.

Amazing performance from Haaland!!

The goalkeeper was absolutely terrible.
```

---

# 📊 Visualizations

The project includes various visualizations:

- Confusion Matrices
- Accuracy Comparison
- Sentiment Distribution
- Model Comparison
- Performance Charts

---

# 🔮 Future Improvements

Potential future enhancements include:

- RoBERTa
- BERT Large
- Live Twitter API Integration
- Live Match Statistics
- Real-Time Sentiment Monitoring
- Explainable AI (SHAP/LIME)
- Multilingual Football Sentiment Analysis
- Live Match Dashboard
- Voice-Based Query System

---

# 👨‍💻 Author

**Hasan**

BS Data Science

Sir Syed University of Engineering & Technology

Karachi, Pakistan

---

# 📄 License

This project has been developed for educational and research purposes.

Feel free to use, modify, and extend the project with proper attribution.

---

# ⭐ If you found this project useful...

Please consider giving it a ⭐ on GitHub.