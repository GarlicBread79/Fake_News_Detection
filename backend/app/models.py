"""Object-oriented classifiers for fake news predictions."""

import joblib
import torch
from transformers import BertForSequenceClassification, BertTokenizer


class LogisticNewsClassifier:
    """Classify news articles with a TF-IDF vectorizer and Logistic Regression."""

    def __init__(
        self,
        vectorizer_path: str = "tfidf_vectorizer.pkl",
        model_path: str = "logistic_regression.pkl",
    ):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> str:
        features = self.vectorizer.transform([text])
        prediction = self.model.predict(features)[0]
        return "Fake News" if prediction == 1 else "Real News"


class BertNewsClassifier:
    """Classify news articles with the fine-tuned BERT model."""

    def __init__(
        self,
        model_path: str = "./bert_finetuned",
        tokenizer_name: str = "bert-base-uncased",
    ):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> str:
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        encoded_input = {
            key: value.to(self.device) for key, value in encoded_input.items()
        }

        with torch.no_grad():
            logits = self.model(**encoded_input).logits

        prediction = torch.argmax(logits, dim=1).item()
        return "Fake News" if prediction == 1 else "Real News"


class FakeNewsDetector:
    """Combine the project's classifiers to return predictions for one article."""

    def __init__(
        self,
        logistic_classifier: LogisticNewsClassifier,
        bert_classifier: BertNewsClassifier,
    ):
        self.logistic_classifier = logistic_classifier
        self.bert_classifier = bert_classifier

    def predict(self, text: str) -> dict[str, str]:
        return {
            "Logistic Regression": self.logistic_classifier.predict(text),
            "BERT": self.bert_classifier.predict(text),
        }
