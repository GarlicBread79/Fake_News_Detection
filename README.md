# Fake News Detection Project

## Overview
This project implements a machine learning-based Fake News Detection System using Logistic Regression and a fine-tuned BERT model. The system classifies news articles as either "Real News"** or "Fake News" based on the text within an article.

## Features
- **TF-IDF + Logistic Regression:** A baseline model using term frequency-inverse document frequency (TF-IDF) features for classification.
- **Fine-Tuned BERT Model:** A deep learning model that improves classification accuracy by leveraging transformer-based contextual embeddings.
- **Model Comparison:** Evaluates both models based on accuracy and ROC-AUC scores.
- **Visualization:** Generates bar charts comparing model performance. 
- **Real-World Testing:** Users can input their own news articles for classification.

## Tech Stack
- **Programming Language:** Python 
- **Machine Learning:** Scikit-Learn, Logistic Regression, TF-IDF
- **Deep Learning:** BERT (Transformer model), PyTorch
- **Visualization:** Matplotlib, Seaborn
- **Data Handling:** Pandas, NumPy

## Installation
1. Clone the repository: git clone https://github.com/Garlicbread79/Fake_News_Detection.git
cd Fake-News-Detection
2. Install required dependencies: pip install -r requirements.txt

## Dataset
- The dataset is sourced from Kaggle and contains labeled news articles.
- It includes features such as article text and labels indicating whether the news is real or fake.

## Model Training
- The Logistic Regression model is trained using TF-IDF features.
- The BERT model is fine-tuned on the dataset using PyTorch's Trainer API.

## Usage
1. Run the Jupyter Notebook (fake_news_detection.ipynb) to train models and analyze results.
2. Use the script to test on real-world news articles: python test.py
- Below is an example of the output from the test file.
**Article:** NASA announces new mission to Mars set for 2030, focusing on human exploration.
**Actual Label:** Real News
Logistic Regression Prediction: Fake News
BERT Prediction: Real News


## Results
| Model                 | Accuracy (%) | ROC-AUC (%) |
|-----------------------|-------------|------------|
| Logistic Regression  | 85.0        | 88.0       |
| BERT (Fine-Tuned)    | 65.0        | 84.0       |


## Evaluation Metrics Explained

**Accuracy** 
Accuracy is defined as the ratio of correct predictions to the total number of predictions. In our project, it measures the overall effectiveness of the model in correctly classifying news articles as either "Real News" or "Fake News". For example, an accuracy of 85% means that the model correctly classifies 85 out of 100 articles. While accuracy provides a straightforward measure of performance, it might not fully capture the model's effectiveness when dealing with imbalanced datasets.

**ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)**
The ROC-AUC metric evaluates how well the model distinguishes between the two classes (real and fake news) across all possible classification thresholds. The ROC curve plots the True Positive Rate (sensitivity) against the False Positive Rate (1 - specificity), and the AUC represents the area under this curve:

AUC = 1: Perfect discrimination between classes.
AUC = 0.5: No discrimination (i.e., the model is no better than random guessing).
AUC = 0: Perfect misclassification, meaning the model's predictions are completely inverted relative to the true labels. An AUC of 0 indicates that there is a critical issue with the model or evaluation process.

By considering both Accuracy and ROC-AUC, we gain a more comprehensive understanding of our models' performance.


## Future Improvements
- Implement additional features such as source credibility and author verification.
- Training on a larger dataset for improved generalization and accuracy.
- Building a web application that allows users to input any article for the system to determine if real or fake. 
- Experiment with ensemble methods or other transformer architectures (e.g., RoBERTa, DistilBERT, ALBERT) to see if they improve performance.
- Use techniques like back translation or synonym replacement to expand the training data and reduce overfitting.
- Develop tools to analyze misclassifications, which can guide further refinements.
- Implement monitoring and logging to track model performance in a production setting.

## Contact
For any questions or contributions, please feel free to reach out.