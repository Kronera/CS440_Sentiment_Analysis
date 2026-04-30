import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    roc_auc_score,)

# Directory to deposit model metrics for display in the app
_METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics")

def _ensure_dir():
    os.makedirs(_METRICS_DIR, exist_ok=True)

# Create and save a confusion matrix for a given model
def _save_confusion_matrix(y_true, y_pred, model_name):
    _ensure_dir()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#f9fafb")
    ax.set_facecolor("white")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Negative", "Positive"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=11)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "#111827")

    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=12, fontweight="bold", color="#111827")
    fig.tight_layout()

    path = os.path.join(_METRICS_DIR, f"{model_name.lower().replace(' ', '_')}_confusion.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path

# Storing model metrics into a JSON
def _save_metrics_json(model_name, accuracy, report_dict):
    _ensure_dir()
    data = {"model": model_name, "accuracy": accuracy, "report": report_dict}
    path = os.path.join(_METRICS_DIR, f"{model_name.lower().replace(' ', '_')}_metrics.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path

# Saving ROC data for display
def roc_data(model_name, fpr, tpr, auc_score):
    _ensure_dir()
    data = {
        "model": model_name,
        "fpr": list(fpr),
        "tpr": list(tpr),
        "auc_score": auc_score,
    }
    path = os.path.join(_METRICS_DIR, f"{model_name.lower().replace(' ', '_')}_roc.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# Load and create the ROC curve graphs
def create_ROCChart():
    from sklearn.metrics import auc as _auc
    _ensure_dir()

    model_keys = [
        ("CNN",           "cnn"),
        ("Naive Bayes",   "naive_bayes"),
        ("Decision Tree", "decision_tree"),
    ]
    colors = ["#4f46e5", "#22c55e", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#f9fafb")
    ax.set_facecolor("white")

    any_plotted = False
    for (display_name, key), color in zip(model_keys, colors):
        roc_path = os.path.join(_METRICS_DIR, key + "_roc.json")
        if not os.path.isfile(roc_path):
            continue
        with open(roc_path) as f:
            data = json.load(f)
        ax.plot(data["fpr"], data["tpr"], color=color, linewidth=2,
                label=f"{display_name}  (AUC = {data['auc_score']:.3f})")
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.plot([0, 1], [0, 1], color="#9ca3af", linewidth=1.2,
            linestyle="--", label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11, color="#374151")
    ax.set_ylabel("True Positive Rate", fontsize=11, color="#374151")
    ax.set_title("ROC Curve — All Models", fontsize=13,
                 fontweight="bold", color="#111827", pad=12)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#6b7280")
    fig.tight_layout()

    out_path = os.path.join(_METRICS_DIR, "roc_combined.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined ROC chart saved to {out_path}")

# Evaluate individual models
def evaluate_model(model, X_test, y_test, model_name="Model") -> None:
    print(f"STEP 5: Evaluating {model_name}")
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"Accuracy : {acc:.4f}\n")
    report = classification_report(y_test, preds,
                                   target_names=["Negative", "Positive"],
                                   output_dict=True)
    print(classification_report(y_test, preds, target_names=["Negative", "Positive"]))

    _save_metrics_json(model_name, acc, report)
    _save_confusion_matrix(y_test, preds, model_name)

    # ROC curve — use predict_proba if available
    try:
        prob_pos = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob_pos)
        auc_score   = roc_auc_score(y_test, prob_pos)
        roc_data(model_name, fpr, tpr, auc_score)
        print(f"AUC: {auc_score:.4f}")
    except Exception as e:
        print(f"Could not compute ROC for {model_name}: {e}")

    print(f"Metrics saved for {model_name}.\n")

# Specialized method to evaluatte CNN
def evaluate_cnn(model, vocab, X_test, y_test, batch_size=64) -> None:
    from models.CNN import ReviewDataset

    model_name = "CNN"
    print(f"Evaluating: {model_name}")

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

    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=["Negative", "Positive"],
                                   output_dict=True)
    print(f"Accuracy : {acc:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

    _save_metrics_json(model_name, acc, report)
    _save_confusion_matrix(all_labels, all_preds, model_name)

    # ROC — raw sigmoid probabilities
    all_probs = []
    with torch.no_grad():
        dataset2 = ReviewDataset(list(X_test), list(y_test), vocab)
        loader2  = DataLoader(dataset2, batch_size=batch_size)
        for X_batch, _ in loader2:
            probs = torch.sigmoid(model(X_batch)).squeeze().tolist()
            if isinstance(probs, float):
                probs = [probs]
            all_probs.extend(probs)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score   = roc_auc_score(all_labels, all_probs)
    roc_data(model_name, fpr, tpr, auc_score)
    print(f"AUC: {auc_score:.4f}")
    print(f"Metrics saved for {model_name}.\n")