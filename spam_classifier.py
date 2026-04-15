"""
SMS Spam Detection Classifier
Classifies SMS messages as spam or legitimate using Naive Bayes and NLP preprocessing.
Dataset: SMS Spam Collection (https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
"""

import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Download NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset(filepath):
    """Load the SMS Spam Collection dataset from a TSV file."""
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["label", "message"],
    )
    print(f"Loaded {len(df)} messages")
    print(f"Duplicates found: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"After removing duplicates: {len(df)} messages")
    return df


# ============================================================
# 2. PREPROCESSING
# ============================================================

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_message(message):
    """Clean and normalize a single SMS message."""
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


def preprocess_dataset(df):
    """Apply preprocessing to all messages in the dataset."""
    print("Preprocessing messages...")
    df["message"] = df["message"].apply(preprocess_message)
    print("Preprocessing complete")
    return df


# ============================================================
# 3. FEATURE EXTRACTION & TRAINING
# ============================================================

def train_model(df):
    """Train a Naive Bayes classifier with hyperparameter tuning."""
    # Convert labels to binary (spam=1, ham=0)
    y = df["label"].apply(lambda x: 1 if x == "spam" else 0)

    # Build pipeline: vectorizer -> classifier
    vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", MultinomialNB())
    ])

    # Hyperparameter tuning with cross-validation
    param_grid = {
        "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")

    print("Training model...")
    grid_search.fit(df["message"], y)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")

    return best_model


# ============================================================
# 4. EVALUATION
# ============================================================

def evaluate_model(model):
    """Test the model on sample messages."""
    test_messages = [
        "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
        "Hey, are we still meeting up for lunch today?",
        "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
        "Reminder: Your appointment is scheduled for tomorrow at 10am.",
        "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
    ]

    processed = [preprocess_message(msg) for msg in test_messages]
    predictions = model.predict(processed)
    probabilities = model.predict_proba(processed)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for i, msg in enumerate(test_messages):
        label = "Spam" if predictions[i] == 1 else "Not Spam"
        spam_prob = probabilities[i][1]
        ham_prob = probabilities[i][0]

        print(f"\nMessage: {msg}")
        print(f"Prediction: {label}")
        print(f"Spam Probability: {spam_prob:.2f}")
        print(f"Not-Spam Probability: {ham_prob:.2f}")
        print("-" * 50)


# ============================================================
# 5. SAVE & LOAD MODEL
# ============================================================

def save_model(model, filename="spam_detection_model.joblib"):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"\nModel saved to {filename}")


def load_model(filename="spam_detection_model.joblib"):
    """Load a saved model from a file."""
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Load and preprocess
    df = load_dataset("sms_spam_collection/SMSSpamCollection")
    df = preprocess_dataset(df)

    # Train
    model = train_model(df)

    # Evaluate
    evaluate_model(model)

    # Save
    save_model(model)
