import gradio as gr
import torch

from preprocessing.cleaner import clean_text
from models.CNN import encode
from main import load_cnn, load_nb, load_tree


# =========================
# LOAD MODELS
# =========================
print("Loading models...")

cnn_model, vocab = load_cnn()
nb_model = load_nb()
tree_model = load_tree()

print("Models loaded.")


# =========================
# CONFIDENCE BAR (FOR UI)
# =========================
def confidence_bar(value):
    percent = int(value * 100)

    # color logic
    if percent >= 70:
        color = "#22c55e"  
    elif percent >= 40:
        color = "#f59e0b"
    else:
        color = "#ef4444" 

    return f"""
    <div style="font-family: sans-serif; margin-top:10px;">
        <div style="margin-bottom:6px;">
            Confidence: {percent}%
        </div>

        <div style="width:100%; background:#e5e7eb; height:14px; border-radius:10px;">
            <div style="
                width:{percent}%;
                height:14px;
                background:{color};
                border-radius:10px;
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """

# =========================
# SENTIMENT SCORE (FOR UI)
# =========================
def sentiment_score(prob_pos):
    return max(-1, min(1, 2 * prob_pos - 1))

# =========================
# MODEL PREDICTIONS
# =========================
def predict_nb(text):
    cleaned = clean_text(text)
    pred = nb_model.predict([cleaned])[0]
    prob = max(nb_model.predict_proba([cleaned])[0])

    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    score = sentiment_score(prob if pred == 1 else -prob)

    return label, confidence_bar(prob), f"Sentiment Score: {score:.2f}"


def predict_tree(text):
    cleaned = clean_text(text)
    pred = tree_model.predict([cleaned])[0]
    prob = max(tree_model.predict_proba([cleaned])[0])

    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    score = sentiment_score(prob if pred == 1 else -prob)

    return label, confidence_bar(prob), f"Sentiment Score: {score:.2f}"


def predict_cnn(text):
    cleaned = clean_text(text)
    encoded = encode(cleaned, vocab)
    x = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        logit = cnn_model(x)
        prob_pos = torch.sigmoid(logit).item()

    label = "POSITIVE" if prob_pos >= 0.5 else "NEGATIVE"
    confidence = max(prob_pos, 1 - prob_pos)

    score = sentiment_score(prob_pos)

    return (
        label,
        confidence_bar(confidence),
        f"Sentiment Score: {score:.2f}"
    )


# =========================
# ROUTER
# =========================
def predict(text, model_name):
    if model_name == "Naive Bayes":
        return predict_nb(text)
    elif model_name == "Decision Tree":
        return predict_tree(text)
    else:
        return predict_cnn(text)


# =========================
# UI
# =========================

with gr.Blocks(theme=gr.themes.Default()) as demo:

    gr.HTML("""
    <style>

        /* FORCE OVERRIDE ALL GRADIO BUTTON STYLES */
        #analyze-btn,
        #analyze-btn.gr-button,
        #analyze-btn.gr-button-primary,
        button#analyze-btn {
            background: #4f46e5 !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            box-shadow: none !important;
        }

        /* Hover state */
        #analyze-btn:hover,
        #analyze-btn.gr-button:hover,
        #analyze-btn.gr-button-primary:hover {
            background: #3730a3 !important;
        }

        /* Active state */
        #analyze-btn:active,
        #analyze-btn.gr-button:active,
        #analyze-btn.gr-button-primary:active {
            background: #312e81 !important;
        }

        /* Remove orange focus ring */
        #analyze-btn:focus {
            outline: none !important;
            box-shadow: none !important;
        }

    </style>
    """)

    gr.Markdown(
        """
        # Text Sentiment Analyzer
        Compare Naive Bayes, Decision Tree, and CNN models
        """
    )

    with gr.Row():

        # INPUT
        with gr.Column():

            gr.Markdown("## Input")

            text_input = gr.Textbox(
                label="Input Text",
                lines=8,
                placeholder="Enter text to analyze sentiment..."
            )

            model_input = gr.Dropdown(
                ["Naive Bayes", "Decision Tree", "CNN"],
                label="Select Model",
                value="CNN"
            )

            btn = gr.Button(
                "Analyze Sentiment",
                elem_id="analyze-btn",
                variant="primary"
            )

        with gr.Column():

            gr.Markdown("## Results")

            output_label = gr.Textbox(
                label="Prediction",
                interactive=False
            )

            output_conf = gr.HTML()

            output_score = gr.Textbox(label="Sentiment Score", interactive=False)

            gr.Markdown("""
            The sentiment score ranges from **-1 to +1**:
            - +1 = very positive
            - 0 = neutral
            - -1 = very negative
            """)

    btn.click(
        fn=predict,
        inputs=[text_input, model_input],
        outputs=[output_label, output_conf, output_score]
    )


demo.launch()