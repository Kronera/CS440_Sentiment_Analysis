# ================================================
#  models/baseline.py — TF-IDF + Logistic Regression
# ================================================

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, LR_MAX_ITER, LR_C


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
        )),
        ("clf", LogisticRegression(
            max_iter=LR_MAX_ITER,
            C=LR_C,
        )),
    ])


def train_model(model: Pipeline, X_train, y_train) -> Pipeline:
    print("Training")

    model.fit(X_train, y_train)

    return model


def build_naive_bayes_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
        )),
        ("clf", MultinomialNB())
    ])

def build_tree_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
        )),
        ("clf", DecisionTreeClassifier(
            random_state=42,
            max_depth=50 
        ))
    ])
