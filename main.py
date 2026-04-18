# ================================================
#  main.py — Entry point
# ================================================

from config import DATA_PATH, SAMPLE_REVIEWS, GLOVE_PATH, GLOVE_DIM, CNN_FREEZE_EPOCHS, CNN_EPOCHS

from data.loader import load_data, split_data
from preprocessing.cleaner import preprocess
from models.baseline import build_model, train_model
from evaluation.metrics import evaluate_cnn, evaluate_model
from predict import predict, predict_cnn
# Nueral Network
from models.CNN import train_cnn
# Bayes Model
from models.baseline import build_naive_bayes_model, train_model
# Decision Tree
from models.baseline import build_tree_model
# Transformer
from models.transformer import train_transformer, evaluate_transformer, predict_transformer


def main():
    # Load csv
    df = load_data(DATA_PATH)

    df = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(df)

    cnn_model, vocab = train_cnn(
        list(X_train), list(y_train),
        list(X_test),  list(y_test),
        epochs=CNN_EPOCHS,
        glove_path=GLOVE_PATH,
        glove_dim=GLOVE_DIM,
        freeze_epochs=CNN_FREEZE_EPOCHS,
    )

    evaluate_cnn(cnn_model, vocab, X_test, y_test)

    predict_cnn(cnn_model, vocab, SAMPLE_REVIEWS)

    train_transformer(X_train, y_train, X_test, y_test)
    evaluate_transformer(X_test, y_test)
    predict_transformer(SAMPLE_REVIEWS)


    # Naive Bayes Model
    print("\n\n=========================")
    print("NAIVE BAYES MODEL")
    print("=========================")

    nb_model = build_naive_bayes_model()
    nb_model = train_model(nb_model, X_train, y_train)

    evaluate_model(nb_model, X_test, y_test)
    predict(nb_model, SAMPLE_REVIEWS)


    # Decision Tree Model
    print("\n\n=========================")
    print("DECISION TREE MODEL")
    print("=========================")

    tree_model = build_tree_model()
    tree_model = train_model(tree_model, X_train, y_train)

    evaluate_model(tree_model, X_test, y_test)
    predict(tree_model, SAMPLE_REVIEWS)


if __name__ == "__main__":
    main()