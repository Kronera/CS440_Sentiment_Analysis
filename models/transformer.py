# ================================================
#  models/transformer.py — DistilBERT fine-tuning & inference
# ================================================
#
#  Requires:  pip install transformers datasets torch accelerate
#
#  What changed vs. the original:
#  - Proper HuggingFace tokenization (not raw character slicing)
#  - Full fine-tuning loop via Trainer API
#  - Separate evaluate_transformer() that prints accuracy + report
#  - predict_transformer() rewritten to use the saved fine-tuned model
#  - Model is saved to disk after training so you don't retrain every run

import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

from config import (
    TRANSFORMER_MODEL,
    TRANSFORMER_MAX_TOKENS,
    TRANSFORMER_SAVE_DIR,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_LR,
    TRANSFORMER_WARMUP_RATIO,
)


# ── Dataset wrapper ──────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    """Wraps tokenized encodings + labels for HuggingFace Trainer."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Tokenisation helper ───────────────────────────────────────────────────────

def _tokenize(texts: list[str], tokenizer) -> dict:
    """
    Tokenize a list of raw strings using the model's own tokenizer.

    Uses truncation=True + padding="max_length" so every sample is exactly
    TRANSFORMER_MAX_TOKENS tokens — no character-slice approximations.
    """
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=TRANSFORMER_MAX_TOKENS,
    )


# ── Metrics for Trainer ───────────────────────────────────────────────────────

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


# ── Training ──────────────────────────────────────────────────────────────────

def train_transformer(X_train, y_train, X_test, y_test) -> None:
    """
    Fine-tune DistilBERT on the provided train split and save the model.

    Args:
        X_train / X_test : pandas Series or list of cleaned review strings
        y_train / y_test : pandas Series or list of int labels (0 = neg, 1 = pos)

    The fine-tuned model is saved to TRANSFORMER_SAVE_DIR (see config.py).
    Call evaluate_transformer() and predict_transformer() after this.
    """
    print("\n\n=========================")
    print("TRANSFORMER (DistilBERT)")
    print("=========================")
    print(f"Base model  : {TRANSFORMER_MODEL}")
    print(f"Max tokens  : {TRANSFORMER_MAX_TOKENS}")
    print(f"Epochs      : {TRANSFORMER_EPOCHS}")
    print(f"Batch size  : {TRANSFORMER_BATCH_SIZE}")
    print(f"Save dir    : {TRANSFORMER_SAVE_DIR}\n")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

    print("Tokenizing ...")
    train_enc = _tokenize(list(X_train), tokenizer)
    test_enc  = _tokenize(list(X_test),  tokenizer)

    train_ds = ReviewDataset(train_enc, list(y_train))
    test_ds  = ReviewDataset(test_enc,  list(y_test))

    # Load DistilBERT with a 2-class classification head
    # ignore_mismatched_sizes silences the expected warning about the new
    # classifier head weights not matching the pretrained checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=TRANSFORMER_SAVE_DIR,
        num_train_epochs=TRANSFORMER_EPOCHS,
        per_device_train_batch_size=TRANSFORMER_BATCH_SIZE,
        per_device_eval_batch_size=TRANSFORMER_BATCH_SIZE * 2,
        learning_rate=TRANSFORMER_LR,
        warmup_ratio=TRANSFORMER_WARMUP_RATIO,   # linear warm-up for first x% of steps
        weight_decay=0.01,                        # L2 regularisation
        eval_strategy="epoch",                    # evaluate at end of every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,              # restore best checkpoint after training
        metric_for_best_model="accuracy",
        logging_steps=100,
        fp16=torch.cuda.is_available(),           # mixed precision if GPU available
        report_to="none",                         # disable W&B / MLflow logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Fine-tuning ...")
    trainer.train()

    # Save the best model + tokenizer together so predict_transformer() can
    # reload them without needing the original base model name
    trainer.save_model(TRANSFORMER_SAVE_DIR)
    tokenizer.save_pretrained(TRANSFORMER_SAVE_DIR)
    print(f"\nModel saved to: {TRANSFORMER_SAVE_DIR}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_transformer(X_test, y_test) -> None:
    """
    Load the fine-tuned model from disk and print accuracy + classification report.
    Must be called after train_transformer().
    """
    if not os.path.isdir(TRANSFORMER_SAVE_DIR):
        print("No saved transformer model found. Run train_transformer() first.")
        return

    print("\nEvaluating Transformer ...")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_SAVE_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_SAVE_DIR)
    model.eval()

    enc     = _tokenize(list(X_test), tokenizer)
    dataset = ReviewDataset(enc, list(y_test))

    trainer = Trainer(model=model, compute_metrics=_compute_metrics)
    results = trainer.predict(dataset)

    preds = np.argmax(results.predictions, axis=1)
    acc   = accuracy_score(y_test, preds)

    print(f"Accuracy : {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=["Negative", "Positive"]))


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_transformer(reviews: list[str]) -> None:
    """
    Load the fine-tuned model from disk and run inference on raw review strings.
    Falls back to the base pretrained model if no fine-tuned model is saved yet.
    """
    model_dir = TRANSFORMER_SAVE_DIR if os.path.isdir(TRANSFORMER_SAVE_DIR) \
                else TRANSFORMER_MODEL

    if model_dir == TRANSFORMER_MODEL:
        print("(No fine-tuned model found — using base pretrained model for inference.)")

    print("=" * 50)
    print("Transformer Predictions on New Reviews")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    for review in reviews:
        # Proper tokenisation — no character slicing
        inputs = tokenizer(
            review,
            truncation=True,
            padding="max_length",
            max_length=TRANSFORMER_MAX_TOKENS,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        probs      = torch.softmax(logits, dim=1).squeeze()
        pred       = torch.argmax(probs).item()
        label      = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = probs[pred].item() * 100

        print(f"Review    : {review[:80]}...")
        print(f"Prediction: {label}  (confidence: {confidence:.1f}%)\n")


