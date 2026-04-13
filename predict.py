# ================================================
#  predict.py — Run inference on new reviews
# ================================================

from preprocessing.cleaner import clean_text


def predict(model, reviews: list[str]) -> None:
    """Run inference on a list of raw review strings."""
    print("=" * 50)
    print("STEP 6: Predictions on New Reviews")
    print("=" * 50)

    cleaned = [clean_text(r) for r in reviews]
    preds   = model.predict(cleaned)
    probs   = model.predict_proba(cleaned)

    for review, pred, prob in zip(reviews, preds, probs):
        label      = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = max(prob) * 100
        print(f"Review    : {review[:80]}...")
        print(f"Prediction: {label}  (confidence: {confidence:.1f}%)\n")
