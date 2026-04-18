# Data
DATA_PATH   = "data\\IMDBDataset.csv"
TEST_SIZE   = 0.2
RANDOM_STATE = 42

# TF-IDF
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE  = (1, 2)

# Logistic Regression
LR_MAX_ITER = 1000
LR_C        = 1.0

# Transformer
TRANSFORMER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
TRANSFORMER_MAX_TOKENS = 512

# Sample reviews for quick inference testing
SAMPLE_REVIEWS = [
    "This film was an absolute masterpiece. The acting was superb and the story kept me hooked throughout.",
    "Terrible movie. Boring plot, bad acting, and a complete waste of two hours. Do not watch.",
    "It was okay, nothing special. Some parts were good but overall pretty forgettable.",
]
