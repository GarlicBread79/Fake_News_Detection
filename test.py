"""
Fake News Detection - Model Testing Script

Run this script to classify a custom news article and compare the models on
predefined test cases.
"""

from backend.app.models import (
    BertNewsClassifier,
    FakeNewsDetector,
    LogisticNewsClassifier,
)


def main() -> None:
    print("Loading Logistic Regression model and TF-IDF vectorizer...")
    logistic_classifier = LogisticNewsClassifier()

    print("Loading fine-tuned BERT model...")
    bert_classifier = BertNewsClassifier()
    detector = FakeNewsDetector(logistic_classifier, bert_classifier)

    input_text = input("Enter the full news article: ")
    predictions = detector.predict(input_text)

    print("\n--- Model Predictions ---")
    for model_name, prediction in predictions.items():
        print(f"{model_name} Prediction: {prediction}")
    print("\n--- End of Predictions ---")

    test_articles = [
        {
            "text": "NASA announces new mission to Mars set for 2030, focusing on human exploration.",
            "label": 0,
        },
        {
            "text": "Breaking: Scientists confirm the Earth is flat, sparking worldwide debate.",
            "label": 1,
        },
    ]

    print("\n--- Testing Model Predictions ---\n")
    for article in test_articles:
        true_label = "Fake News" if article["label"] == 1 else "Real News"
        article_predictions = detector.predict(article["text"])

        print(f"\n**Article:** {article['text']}")
        print(f"**Actual Label:** {true_label}")
        for model_name, prediction in article_predictions.items():
            print(f"{model_name} Prediction: {prediction}")

    print("\n--- Testing Completed ---")


if __name__ == "__main__":
    main()
