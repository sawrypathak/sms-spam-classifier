# sms-spam-classifier
A Python ML model to classify SMS messages as spam or ham using TF-IDF and classifiers.

## 🧠 Overview

This project uses the famous **SMS Spam Collection Dataset** to train and evaluate models that can distinguish spam from legitimate messages.

Techniques used:
- Text preprocessing
- TF-IDF vectorization
- Supervised classification (Naive Bayes, Logistic Regression, SVM)
- Model evaluation (accuracy, precision, recall, F1-score)

## 🗃️ Dataset

The dataset `spam.csv` contains two columns:
- `v1`: label (ham/spam)
- `v2`: message text

It's originally from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/sawrypathak/sms-spam-classifier.git
cd sms-spam-classifier
