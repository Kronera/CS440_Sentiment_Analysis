# Baseline models for the NLP

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, LR_MAX_ITER, LR_C, \
                   GB_N_ESTIMATORS, GB_MAX_DEPTH, GB_LEARNING_RATE, GB_SUBSAMPLE


# Build models into the NLP pipeline
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
        ("clf", ComplementNB(alpha=0.1))
    ])

def build_tree_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
        )),
        ("clf", GradientBoostingClassifier(
            n_estimators=GB_N_ESTIMATORS,
            max_depth=GB_MAX_DEPTH,
            learning_rate=GB_LEARNING_RATE,
            subsample=GB_SUBSAMPLE,
            max_features="sqrt",
            random_state=42,
            verbose=1,
        ))
    ])
