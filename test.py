"""
Fake News Detection - Model Testing Script

This script loads pre-trained models (Logistic Regression + TF-IDF and Fine-Tuned BERT)
to classify news articles as either "Real News" or "Fake News."

Features:
- Takes user input for testing a custom article.
- Compares predictions from Logistic Regression and BERT.
- Runs predefined test cases for benchmarking.
"""

import pandas as pd
import joblib
import torch
import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification

# ================================
# Step 1 - Load Pre-trained Models
# ================================

print("Loading Logistic Regression model and TF-IDF vectorizer...")

# Loads the trained Logistic Regression model and TF-IDF vectorizer
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
logistic_model = joblib.load("logistic_regression.pkl")

print("Loading fine-tuned BERT model...")

# Loads a fine-tuned BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("./bert_finetuned")

# Moves the BERT model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()  

# ================================
# Step 2 - Define Prediction Functions
# ================================

def predict_logistic(text: str) -> str:
    """
    Predicts whether a given news article is real or fake using the Logistic Regression model.

    Args:
        text (str): The news article text.

    Returns:
        str: "Real News" or "Fake News" based on the model's prediction.
    """
    text_tfidf = tfidf_vectorizer.transform([text])  # Converts the text to TF-IDF features
    pred = logistic_model.predict(text_tfidf)[0]  # Makes a prediction (0: Real, 1: Fake)
    return "Fake News" if pred == 1 else "Real News"


def predict_bert(text: str) -> str:
    """
    Predicts whether a given news article is real or fake using the fine-tuned BERT model.

    Args:
        text (str): The news article text.

    Returns:
        str: "Real News" or "Fake News" based on the model's prediction.
    """
    encoded_input = bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}  # 

    with torch.no_grad():  
        logits = bert_model(**encoded_input).logits

    pred = torch.argmax(logits, dim=1).item()  # Gets the predicted label (0: Real, 1: Fake)
    return "Fake News" if pred == 1 else "Real News"


# ================================
# Step 3 - User Input for Custom Prediction
# ================================

input_text = input("Enter the full news article: ")

# Makes a prediction using both models
logistic_prediction = predict_logistic(input_text)
bert_prediction = predict_bert(input_text)

# Displays the Results
print("\n--- Model Predictions ---")
print(f"Logistic Regression Prediction: {logistic_prediction}")
print(f"BERT Prediction: {bert_prediction}")
print("\n--- End of Predictions ---")


# ================================
# Step 4 - Predefined Test Cases for Benchmarking
# ================================

# Articles to test the models
test_articles = [
    {"text": "NASA announces new mission to Mars set for 2030, focusing on human exploration.", "label": 0},  # Real
    {"text": "Breaking: Scientists confirm the Earth is flat, sparking worldwide debate.", "label": 1},  # Fake
]

# Runs the Predictions and Compare Results
print("\n--- Testing Model Predictions ---\n")
for article in test_articles:
    true_label = "Fake News" if article["label"] == 1 else "Real News"
    logistic_pred = predict_logistic(article["text"])
    bert_pred = predict_bert(article["text"])

    print(f"\n**Article:** {article['text']}")
    print(f"**Actual Label:** {true_label}")
    print(f"Logistic Regression Prediction: {logistic_pred}")
    print(f"BERT Prediction: {bert_pred}")

print("\n--- Testing Completed ---")
