import torch
from preprocessing.cleaner import clean_text
from models.CNN import encode


def predict(model, reviews: list[str]) -> None:
    """Run inference on a list of raw review strings using an sklearn pipeline model."""
    print("=" * 50)
    print("Predictions on New Reviews")
    print("=" * 50)

    cleaned = [clean_text(r) for r in reviews]
    preds   = model.predict(cleaned)
    probs   = model.predict_proba(cleaned)

    for review, pred, prob in zip(reviews, preds, probs):
        label      = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = max(prob) * 100
        print(f"Review    : {review[:80]}...")
        print(f"Prediction: {label}  (confidence: {confidence:.1f}%)\n")


def predict_cnn(model, vocab: dict, reviews: list[str]) -> None:
    """Run inference on a list of raw review strings using the PyTorch CNN model."""
    print("=" * 50)
    print("CNN Predictions on New Reviews")
    print("=" * 50)

    model.eval()

    cleaned = [clean_text(r) for r in reviews]

    with torch.no_grad():
        for review, cleaned_review in zip(reviews, cleaned):
            # Encode the review into a tensor and add a batch dimension
            encoded = encode(cleaned_review, vocab)
            x       = torch.tensor([encoded], dtype=torch.long)

            # Forward pass — raw logit → sigmoid → probability
            logit      = model(x)
            prob_pos   = torch.sigmoid(logit).item()
            prob_neg   = 1.0 - prob_pos

            label      = "POSITIVE" if prob_pos >= 0.5 else "NEGATIVE"
            confidence = max(prob_pos, prob_neg) * 100

            print(f"Review    : {review[:80]}...")
            print(f"Prediction: {label}  (confidence: {confidence:.1f}%)\n")

