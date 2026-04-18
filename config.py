# ================================================
#  config.py — Central configuration
# ================================================

# Data
DATA_PATH   = "data/IMDBDataset.csv"
TEST_SIZE   = 0.2
RANDOM_STATE = 42

# GloVe Embeddings
# Download from: https://nlp.stanford.edu/projects/glove/  (glove.6B.zip)
# Then point this path at the extracted .txt file, e.g. "glove.6B.100d.txt"
# Set to None to fall back to random embeddings (original behaviour).
GLOVE_PATH = "data/glove.6B.100d.txt"   # <-- update this to your actual path
GLOVE_DIM  = 100                    # must match the file: 50 / 100 / 200 / 300
CNN_FREEZE_EPOCHS = 3               # epochs to keep embeddings frozen before fine-tuning
CNN_EPOCHS        = 10              # more epochs needed when using pretrained embeddings

# TF-IDF
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE  = (1, 2)

# Logistic Regression
LR_MAX_ITER = 1000
LR_C        = 1.0

# Gradient Boosting (replaces single Decision Tree)
# n_estimators: more trees = better accuracy but slower training.
#   200 is a good balance; push to 500 if you have time.
# max_depth=4: shallow trees prevent overfitting on sparse TF-IDF features.
# learning_rate=0.1: standard starting point; lower (0.05) + more trees is
#   often better but slower.
# subsample=0.8: stochastic boosting — each tree sees 80% of samples.
#
# LightGBM alternative (10-20x faster, same or better accuracy):
#   pip install lightgbm
#   from lightgbm import LGBMClassifier
#   LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
#                  subsample=0.8, colsample_bytree=0.5, random_state=42)
GB_N_ESTIMATORS  = 200
GB_MAX_DEPTH     = 4
GB_LEARNING_RATE = 0.1
GB_SUBSAMPLE     = 0.8

# Transformer (DistilBERT fine-tuning)
#
# TRANSFORMER_MODEL: base checkpoint to fine-tune from.
#   "distilbert-base-uncased" — fine-tune from scratch on your data (recommended
#     when your domain differs from SST-2, e.g. Amazon reviews).
#   "distilbert-base-uncased-finetuned-sst-2-english" — start from an already
#     sentiment-tuned checkpoint; converges faster on IMDB but may need more
#     epochs to adapt to other domains.
#
# TRANSFORMER_SAVE_DIR: where the fine-tuned model + tokenizer are saved.
#   After the first run you can skip retraining and just call
#   evaluate_transformer() / predict_transformer() directly.
#
# TRANSFORMER_EPOCHS: 3 is usually enough for DistilBERT on IMDB.
#   EarlyStoppingCallback will halt if val accuracy stops improving.
#
# TRANSFORMER_BATCH_SIZE: 16 fits comfortably in 6GB VRAM.
#   Drop to 8 if you get OOM errors; increase to 32 on a bigger GPU.
#
# TRANSFORMER_LR: 2e-5 is the standard fine-tuning learning rate for BERT-family
#   models. Too high (>5e-5) causes catastrophic forgetting of pretrained weights.
#
# TRANSFORMER_WARMUP_RATIO: linearly ramps LR from 0 → TRANSFORMER_LR over the
#   first 6% of training steps, then decays. Helps stabilise early training.
TRANSFORMER_MODEL       = "distilbert-base-uncased"
TRANSFORMER_MAX_TOKENS  = 512
TRANSFORMER_SAVE_DIR    = "saved_transformer"
TRANSFORMER_EPOCHS      = 3
TRANSFORMER_BATCH_SIZE  = 16
TRANSFORMER_LR          = 2e-5
TRANSFORMER_WARMUP_RATIO = 0.06

# Sample reviews for quick inference testing
SAMPLE_REVIEWS = [
    "This film was an absolute masterpiece. The acting was superb and the story kept me hooked throughout.",
    "Terrible movie. Boring plot, bad acting, and a complete waste of two hours. Do not watch.",
    "It was okay, nothing special. Some parts were good but overall pretty forgettable.",
]
