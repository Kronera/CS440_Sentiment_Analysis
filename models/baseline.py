# ================================================
#  models/baseline.py — TF-IDF + Logistic Regression
# ================================================

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
