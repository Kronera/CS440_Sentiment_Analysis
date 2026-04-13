from config import TRANSFORMER_MODEL, TRANSFORMER_MAX_TOKENS


def predict_transformer(reviews: list[str]) -> None:

    print("Transformer")

    from transformers import pipeline as hf_pipeline

    classifier = hf_pipeline(
        model=TRANSFORMER_MODEL,
    )

    for review in reviews:
        result = classifier(review[:TRANSFORMER_MAX_TOKENS])[0]
        print(f"Review    : {review[:80]}...")
        print(f"Prediction: {result['label']}  (score: {result['score']:.4f})\n")


