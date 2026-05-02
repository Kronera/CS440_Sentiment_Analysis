# Demo Link: https://colab.research.google.com/drive/1BD0nGiqHX2cGL_fpNEEeo7ahobfJCU_O?usp=sharing

# CS440 
# Yelp Business Sentiment Analyzer

A Natural Language Processing (NLP) project that analyzes Yelp reviews and predicts sentiment using multiple machine learning models. The system includes an interactive web app for exploring sentiment across businesses and individual reviews.

---

## Features

* Multiple Models

  * Convolutional Neural Network (CNN) with GloVe embeddings
  * Naive Bayes classifier
  * Decision Tree classifier

* Evaluation Metrics

  * Accuracy, Precision, Recall, F1-score
  * Confusion matrices
  * ROC curve comparison across models

* Interactive Web App (Gradio)

  * Browse Yelp businesses
  * Analyze all reviews for a business
  * Navigate individual reviews
  * Highlight important sentiment-driving words
  * Visual dashboards (distribution, confidence, etc.)

---

## Setup Instructions

### 1. Download Required Data

Download the dataset and embeddings from Google Drive:

https://drive.google.com/file/d/1IgU2ovgNNhR5_IzDdgdfLphw_zSFsGBz/view?usp=sharing

Extract the contents into:

```
CS440_Sentiment_Analysis/data/
```

This should include:

* Yelp review dataset (`yelp_review.JSON`)
* Yelp business dataset (`yelp_business.JSON`)
* GloVe embeddings

---

### 2. Install Dependencies

Install required packages:

```bash
pip install torch scikit-learn pandas numpy matplotlib gradio joblib
```
---

## Running the Project

### Step 1: Train Models

```bash
python3 main.py
```

This will:

* Load and preprocess data
* Train all models (CNN, Naive Bayes, Decision Tree)
* Save models:

  * `cnn_model.pt`
  * `nb_model.pkl`
  * `tree_model.pkl`
* Generate evaluation metrics and charts

---

### Step 2: Launch the App

```bash
python3 app.py
```

Then open:

```
http://127.0.0.1:7860/
```

---

## Using the App

### Business Analysis

* Browse businesses using navigation buttons
* Click "Analyze Selected Business"
* View:

  * % positive vs negative reviews
  * Sentiment score
  * Confidence levels
  * Visual charts

### Review Explorer

* Navigate individual reviews
* See:

  * Predicted sentiment
  * Confidence score
  * Highlighted keywords
  * Star ratings
  * Ambiguity detection (model disagreement)

---

## Models Overview

### CNN (Deep Learning)

* Uses pretrained GloVe embeddings
* Captures contextual meaning of words
* Best overall performance

### Naive Bayes

* Fast baseline model
* Uses TF-IDF features
* Provides keyword importance (used for highlighting)

### Decision Tree

* Interpretable model
* Captures non-linear patterns

---

## Evaluation

Each model is evaluated using:

* Accuracy
* Precision / Recall / F1-score
* Confusion Matrix
* ROC Curve (combined visualization)

Metrics are saved in:

```
metrics/
```

---

## Notes

* The dataset is large → training may take time
* By default, training uses a sample of 3000 reviews for speed
  (see `main.py` → `df.sample(3000)`)

To train on full dataset:

```python
# Comment this line in main.py
df = df.sample(3000, random_state=42)
```

---
