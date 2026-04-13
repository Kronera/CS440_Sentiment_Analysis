# ================================================
#  evaluation/metrics.py — Evaluate model performance
# ================================================

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)


def evaluate_model(model, X_test, y_test) -> None:
    print("STEP 5: Evaluating")
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"Accuracy : {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=["Negative", "Positive"]))
    _plot_confusion_matrix(y_test, preds)

# CNN evaluation is more complex due to PyTorch's data handling, so we have a separate function for it
def evaluate_cnn(model, vocab, X_test, y_test, batch_size=64) -> None:
    from models.CNN import ReviewDataset

    print("Neural Network (CNN)")

    # Building dataloader from test set
    dataset = ReviewDataset(list(X_test), list(y_test), vocab)
    loader  = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = torch.sigmoid(model(X_batch))
            preds   = (outputs > 0.5).long()
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.long().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy : {acc:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

    _plot_confusion_matrix(all_labels, all_preds)

# Plotting confusion matrixm, this might be helpful going forward
def _plot_confusion_matrix(y_true, y_pred) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()