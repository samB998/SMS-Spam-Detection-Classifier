# SMS-Spam-Detection-Classifier
SMS spam detection model using Naive Bayes, NLTK, and scikit-learn with hyperparameter tuning and cross-validation.

# SMS Spam Detection Classifier

A machine learning model that classifies SMS messages as spam or legitimate using Natural Language Processing.

## Overview

This project builds a Naive Bayes classifier trained on the SMS Spam Collection dataset (5,574 messages). It processes raw text through a full ML pipeline — from preprocessing to deployment.

## How It Works

1. **Data Loading** — Load and clean the SMS Spam Collection dataset
2. **Preprocessing** — Lowercase, remove punctuation, tokenize, remove stop words, stem words
3. **Feature Extraction** — Convert text to numerical vectors using CountVectorizer (unigrams + bigrams)
4. **Training** — Train a Multinomial Naive Bayes classifier with GridSearchCV hyperparameter tuning
5. **Evaluation** — Evaluate on unseen messages with prediction probabilities
6. **Deployment** — Save and load the model using joblib

## Tech Stack

- Python
- scikit-learn (Naive Bayes, Pipeline, GridSearchCV)
- NLTK (tokenization, stemming, stop words)
- pandas
- joblib

## Results

- Optimized using 5-fold cross-validation with F1 score
- High accuracy distinguishing spam from legitimate messages
- Outputs confidence probabilities for each prediction

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,574 SMS messages labeled as ham or spam.
