import os
import joblib
import torch

from config import DATA_PATH, SAMPLE_REVIEWS, GLOVE_PATH, GLOVE_DIM, CNN_FREEZE_EPOCHS, CNN_EPOCHS

from data.loader import load_data, split_data
from preprocessing.cleaner import preprocess
from evaluation.metrics import evaluate_cnn, evaluate_model, create_ROCChart
from predict import predict, predict_cnn

from models.CNN import train_cnn, TextCNN
from models.baseline import build_naive_bayes_model, build_tree_model, train_model

TRAIN_MODE = True 



CNN_PATH  = "cnn_model.pt"
NB_PATH   = "nb_model.pkl"
TREE_PATH = "tree_model.pkl"

def train_all():
    print("Training all models\n")

    df = load_data(DATA_PATH)

    # Small sample for faster training. Comment out for full dataset.
    df = df.sample(3000, random_state=42)

    df = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(df)

    # CNN
    cnn_model, vocab = train_cnn(
        list(X_train), list(y_train),
        list(X_test),  list(y_test),
        epochs=CNN_EPOCHS,
        glove_path=GLOVE_PATH,
        glove_dim=GLOVE_DIM,
        freeze_epochs=CNN_FREEZE_EPOCHS,
    )

    torch.save({
        "model_state": cnn_model.state_dict(),
        "vocab": vocab,
        "vocab_size": len(vocab)
    }, CNN_PATH)

    evaluate_cnn(cnn_model, vocab, X_test, y_test)
    predict_cnn(cnn_model, vocab, SAMPLE_REVIEWS)





    # Naive Bayes
    print("\nNAIVE BAYES MODEL\n")

    nb_model = build_naive_bayes_model()
    nb_model = train_model(nb_model, X_train, y_train)

    joblib.dump(nb_model, NB_PATH)

    evaluate_model(nb_model, X_test, y_test, model_name="Naive Bayes")
    predict(nb_model, SAMPLE_REVIEWS)


    # Gradient Boost
    print("\nTREE MODEL\n")

    tree_model = build_tree_model()
    tree_model = train_model(tree_model, X_train, y_train)

    joblib.dump(tree_model, TREE_PATH)

    evaluate_model(tree_model, X_test, y_test, model_name="Decision Tree")
    predict(tree_model, SAMPLE_REVIEWS)
    create_ROCChart()


#L Loading app GUI 
def load_cnn():
    checkpoint = torch.load(CNN_PATH)
    vocab = checkpoint["vocab"]
    vocab_size = checkpoint["vocab_size"]

    model = TextCNN(vocab_size)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, vocab


def load_nb():
    return joblib.load(NB_PATH)


def load_tree():
    return joblib.load(TREE_PATH)


def main():
    if TRAIN_MODE or not (
        os.path.exists(CNN_PATH)
        and os.path.exists(NB_PATH)
        and os.path.exists(TREE_PATH)
    ):
        train_all()
    else:
        print("Models already trained. Loading...\n")

        cnn_model, vocab = load_cnn()
        nb_model = load_nb()
        tree_model = load_tree()

        predict_cnn(cnn_model, vocab, SAMPLE_REVIEWS)
        predict(nb_model, SAMPLE_REVIEWS)
        predict(tree_model, SAMPLE_REVIEWS)


if __name__ == "__main__":
    main()