# Trainign and test data (80/20 split)
DATA_PATH   = "data/IMDBDataset.csv"
TEST_SIZE   = 0.2
RANDOM_STATE = 42

# GloVe Embeddings
# Download from: https://nlp.stanford.edu/projects/glove/  (glove.6B.zip)
# more epochs needed when using pretrained embeddings
GLOVE_PATH = "data/glove.6B.100d.txt"
GLOVE_DIM  = 100 
CNN_FREEZE_EPOCHS = 3
CNN_EPOCHS = 10 

# TF-IDF
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE  = (1, 2)

# Gradient Boosting (Decision Tree)
# 200 trees is a good for the project
# max_depth=4: shallow trees prevent overfitting on sparse TF-IDF features.
# LightGBM alternative (10-20x faster, same or better accuracy):
# pip install lightgbm
# subsample=0.8, colsample_bytree=0.5, random_state=42)
GB_N_ESTIMATORS  = 200
GB_MAX_DEPTH     = 4
GB_LEARNING_RATE = 0.1
GB_SUBSAMPLE     = 0.8
