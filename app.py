import json
import os
import re

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from preprocessing.cleaner import clean_text
from models.CNN import encode
from main import load_cnn, load_nb, load_tree


# =========================
# LOAD MODELS
# =========================
print("Loading models...")
cnn_model, vocab = load_cnn()
nb_model         = load_nb()
tree_model       = load_tree()
print("Models loaded.")



# =========================
# METRICS LOADER
# =========================
_METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")


def _fmt(val, decimals=3):
    return f"{val:.{decimals}f}"


def _load_metrics_html():
    import json as _json
    import base64

    if not os.path.isdir(_METRICS_DIR):
        return "<p style='color:#9ca3af;font-family:sans-serif;'>No metrics found. Run <code>main.py</code> to generate metrics.</p>"

    models = [
        ("CNN",           "cnn"),
        ("Naive Bayes",   "naive_bayes"),
        ("Decision Tree", "decision_tree"),
    ]

    html = ""
    for display_name, key in models:
        json_path = os.path.join(_METRICS_DIR, key + "_metrics.json")
        img_path  = os.path.join(_METRICS_DIR, key + "_confusion.png")

        if not os.path.isfile(json_path):
            continue

        with open(json_path) as f:
            data = _json.load(f)

        acc    = data["accuracy"]
        report = data["report"]
        neg    = report.get("Negative", {})
        pos    = report.get("Positive", {})
        macro  = report.get("macro avg", {})

        img_tag = ""
        if os.path.isfile(img_path):
            with open(img_path, "rb") as imgf:
                b64 = base64.b64encode(imgf.read()).decode()
            img_tag = '<img src="data:image/png;base64,' + b64 + '" style="max-width:340px;border-radius:8px;border:1px solid #e5e7eb;" />'

        acc_color = "#22c55e" if acc >= 0.8 else "#f59e0b" if acc >= 0.6 else "#ef4444"

        def row(cls, vals):
            return (
                "<tr style='border-bottom:1px solid #f3f4f6;'>"
                "<td style='padding:8px;font-weight:600;color:#374151;'>" + cls + "</td>"
                "<td style='padding:8px;text-align:center;color:#374151;'>" + _fmt(vals.get("precision", 0)) + "</td>"
                "<td style='padding:8px;text-align:center;color:#374151;'>" + _fmt(vals.get("recall", 0)) + "</td>"
                "<td style='padding:8px;text-align:center;color:#374151;'>" + _fmt(vals.get("f1-score", 0)) + "</td>"
                "<td style='padding:8px;text-align:center;color:#6b7280;'>" + str(int(vals.get("support", 0))) + "</td>"
                "</tr>"
            )

        macro_row = (
            "<tr style='background:#f9fafb;'>"
            "<td style='padding:8px;font-weight:600;color:#374151;'>Macro Avg</td>"
            "<td style='padding:8px;text-align:center;color:#374151;'>" + _fmt(macro.get("precision", 0)) + "</td>"
            "<td style='padding:8px;text-align:center;color:#374151;'>" + _fmt(macro.get("recall", 0)) + "</td>"
            "<td style='padding:8px;text-align:center;font-weight:700;color:#4f46e5;'>" + _fmt(macro.get("f1-score", 0)) + "</td>"
            "<td style='padding:8px;'></td>"
            "</tr>"
        )

        html += (
            "<div style='margin-bottom:24px;padding:16px;background:white;"
            "border-radius:12px;border:1px solid #e5e7eb;'>"
            "<h3 style='margin:0 0 14px 0;color:#111827;font-family:sans-serif;'>"
            + display_name +
            "<span style='background:" + acc_color + ";color:white;font-size:0.72em;"
            "padding:2px 10px;border-radius:12px;margin-left:10px;"
            "vertical-align:middle;'>Accuracy " + _fmt(acc * 100, 1) + "%</span>"
            "</h3>"
            "<div style='display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;'>"
            "<div style='flex:1;min-width:260px;'>"
            "<table style='width:100%;border-collapse:collapse;font-family:sans-serif;font-size:0.88em;'>"
            "<thead><tr style='background:#f3f4f6;'>"
            "<th style='padding:8px;text-align:left;color:#6b7280;'>Class</th>"
            "<th style='padding:8px;text-align:center;color:#6b7280;'>Precision</th>"
            "<th style='padding:8px;text-align:center;color:#6b7280;'>Recall</th>"
            "<th style='padding:8px;text-align:center;color:#6b7280;'>F1</th>"
            "<th style='padding:8px;text-align:center;color:#6b7280;'>Support</th>"
            "</tr></thead>"
            "<tbody>"
            + row("Negative", neg)
            + row("Positive", pos)
            + macro_row +
            "</tbody></table></div>"
            "<div style='flex:0 0 auto;'>" + img_tag + "</div>"
            "</div></div>"
        )

    # Add combined ROC chart at the top
    roc_path = os.path.join(_METRICS_DIR, "roc_combined.png")
    roc_html = ""
    if os.path.isfile(roc_path):
        with open(roc_path, "rb") as imgf:
            b64 = base64.b64encode(imgf.read()).decode()
        roc_html = (
            "<div style='margin-bottom:24px;padding:16px;background:white;"
            "border-radius:12px;border:1px solid #e5e7eb;'>"
            "<h3 style='margin:0 0 12px 0;color:#111827;font-family:sans-serif;'>ROC Curve — All Models</h3>"
            "<img src='data:image/png;base64," + b64 + "' style='max-width:520px;border-radius:8px;border:1px solid #e5e7eb;' />"
            "<p style='margin:10px 0 0 0;font-family:sans-serif;font-size:0.85em;color:#6b7280;'>"
            "</p></div>"
        )

    final = roc_html + html
    return final if final.strip() else "<p style='color:#9ca3af;'>No metrics files found in the metrics/ folder.</p>"

# =========================
# NAIVE BAYES KEYWORD ENGINE
# =========================
def _get_nb_word_scores():
    feature_names = nb_model.named_steps["tfidf"].get_feature_names_out()
    log_probs     = nb_model.named_steps["clf"].feature_log_prob_
    scores        = log_probs[1] - log_probs[0]
    return dict(zip(feature_names, scores))

_NB_SCORES = _get_nb_word_scores()


def _highlight_review(raw_text):
    cleaned_tokens = clean_text(raw_text).split()
    token_scores = {}
    for tok in cleaned_tokens:
        score = _NB_SCORES.get(tok)
        if score is not None:
            base    = tok.replace("_NEG", "").lower()
            flipped = -score if tok.endswith("_NEG") else score
            if base not in token_scores or abs(flipped) > abs(token_scores[base]):
                token_scores[base] = flipped
    if not token_scores:
        return f"<span>{raw_text}</span>"
    max_abs = max(abs(v) for v in token_scores.values()) or 1

    def _word_html(word):
        key   = re.sub(r"[^a-z]", "", word.lower())
        score = token_scores.get(key)
        if score is None:
            return f'<span style="color:#374151;">{word}</span>'
        intensity = min(abs(score) / max_abs, 1.0)
        if score > 0:
            r, g, b = int(34+(1-intensity)*200), int(197-(1-intensity)*80), int(94+(1-intensity)*100)
        else:
            r, g, b = int(239-(1-intensity)*80), int(68+(1-intensity)*150), int(68+(1-intensity)*150)
        bg     = f"rgba({r},{g},{b},{0.15+intensity*0.35})"
        border = f"rgba({r},{g},{b},0.6)"
        title  = f"{'positive' if score>0 else 'negative'} signal ({score:+.2f})"
        return (f'<span title="{title}" style="background:{bg};border-bottom:2px solid {border};'
                f'border-radius:3px;padding:1px 2px;color:#111827;">{word}</span>')

    parts = re.findall(r"\w+|[^\w]", raw_text)
    return "".join(_word_html(p) if re.match(r"\w+", p) else p for p in parts)


def _top_keywords(raw_text, n=5):
    cleaned_tokens = clean_text(raw_text).split()
    found = {}
    for tok in cleaned_tokens:
        score = _NB_SCORES.get(tok)
        if score is not None:
            base    = tok.replace("_NEG", "")
            flipped = -score if tok.endswith("_NEG") else score
            if base not in found or abs(flipped) > abs(found[base]):
                found[base] = flipped
    sorted_words = sorted(found.items(), key=lambda x: x[1], reverse=True)
    return [(w,s) for w,s in sorted_words if s>0][:n], [(w,s) for w,s in sorted_words if s<0][-n:][::-1]


def _keywords_html(top_pos, top_neg):
    def pill(word, score, color):
        intensity = min(abs(score), 3) / 3
        opacity   = 0.2 + intensity * 0.5
        return (f'<span style="display:inline-block;background:rgba({color},{opacity});'
                f'border-radius:12px;padding:2px 10px;margin:2px;font-size:0.85em;'
                f'color:#111827;font-weight:500;">{word}</span>')
    pos_pills = "".join(pill(w,s,"34,197,94") for w,s in top_pos) or "<em style='color:#9ca3af'>none found</em>"
    neg_pills = "".join(pill(w,s,"239,68,68") for w,s in top_neg) or "<em style='color:#9ca3af'>none found</em>"
    return f"""
    <div style="margin-top:14px;font-family:sans-serif;">
      <div style="font-size:0.8em;color:#6b7280;font-weight:600;
                  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Positive signals</div>
      <div style="margin-bottom:10px;">{pos_pills}</div>
      <div style="font-size:0.8em;color:#6b7280;font-weight:600;
                  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Negative signals</div>
      <div>{neg_pills}</div>
    </div>"""


# =========================
# SHARED HELPERS
# =========================
def sentiment_score(prob_pos):
    return max(-1, min(1, 2 * prob_pos - 1))


def _run_cnn(text):
    cleaned  = clean_text(text)
    encoded  = encode(cleaned, vocab)
    x        = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        prob_pos = torch.sigmoid(cnn_model(x)).item()
    return ("POSITIVE" if prob_pos >= 0.5 else "NEGATIVE"), prob_pos


def _run_model(text, model_name):
    cleaned = clean_text(text)
    if model_name == "Naive Bayes":
        pred     = nb_model.predict([cleaned])[0]
        prob_pos = nb_model.predict_proba([cleaned])[0][1]
    elif model_name == "Decision Tree":
        pred     = tree_model.predict([cleaned])[0]
        prob_pos = tree_model.predict_proba([cleaned])[0][1]
    else:
        encoded  = encode(cleaned, vocab)
        x        = torch.tensor([encoded], dtype=torch.long)
        with torch.no_grad():
            prob_pos = torch.sigmoid(cnn_model(x)).item()
        pred = 1 if prob_pos >= 0.5 else 0
    return ("POSITIVE" if pred == 1 else "NEGATIVE"), prob_pos


def _review_card_html(label, prob_pos, stars, highlighted_text, keywords_html, disagrees=False):
    confidence  = max(prob_pos, 1 - prob_pos)
    score       = sentiment_score(prob_pos)
    conf_pct    = int(confidence * 100)
    label_bg    = "#22c55e" if label == "POSITIVE" else "#ef4444"
    bar_color   = "#22c55e" if conf_pct >= 70 else "#f59e0b" if conf_pct >= 40 else "#ef4444"
    gauge_pct   = int((score + 1) / 2 * 100)
    gauge_color = "#22c55e" if score >= 0.2 else "#ef4444" if score <= -0.2 else "#f59e0b"
    star_str       = "★" * int(stars) + "☆" * (5 - int(stars)) if stars else "—"
    disagree_badge = (
        '<span title="CNN and Decision Tree disagree on this review — may contain mixed sentiment" '
        'style="background:#fef3c7;color:#92400e;border:1px solid #fcd34d;font-size:0.8em;'
        'font-weight:600;padding:3px 10px;border-radius:12px;">⚠ Mixed Signals</span>'
        if disagrees else ""
    )
    return f"""
    <div style="font-family:sans-serif;">
      <div style="padding:16px;background:#f9fafb;border-radius:14px;
                  border:1px solid #e5e7eb;margin-bottom:12px;">
        <div style="font-size:0.75em;color:#6b7280;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">CNN · Sentiment</div>
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
          <span style="background:{label_bg};color:white;font-size:1.05em;
                       font-weight:700;padding:5px 18px;border-radius:20px;">{label}</span>
          <span style="font-size:1.2em;color:#f59e0b;letter-spacing:2px;">{star_str}</span>
          {disagree_badge}
        </div>
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;
                      font-size:0.82em;color:#6b7280;">
            <span>Confidence</span><span style="font-weight:600;">{conf_pct}%</span>
          </div>
          <div style="width:100%;background:#e5e7eb;height:10px;border-radius:8px;">
            <div style="width:{conf_pct}%;height:10px;background:{bar_color};border-radius:8px;"></div>
          </div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;
                      font-size:0.82em;color:#6b7280;">
            <span>Sentiment Score</span>
            <span style="font-weight:700;color:{gauge_color};">{score:+.2f}</span>
          </div>
          <div style="position:relative;width:100%;background:#e5e7eb;height:10px;border-radius:8px;">
            <div style="position:absolute;left:50%;top:0;width:2px;height:10px;background:#9ca3af;"></div>
            <div style="width:{gauge_pct}%;height:10px;background:{gauge_color};border-radius:8px;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.72em;
                      color:#9ca3af;margin-top:2px;">
            <span>-1</span><span>0</span><span>+1</span>
          </div>
        </div>
      </div>
      <div style="padding:16px;background:#f9fafb;border-radius:14px;border:1px solid #e5e7eb;">
        <div style="font-size:0.75em;color:#6b7280;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">
          Naive Bayes · Key Signals</div>
        <div style="font-size:0.78em;color:#9ca3af;margin-bottom:8px;">
          Words highlighted by sentiment strength · hover for score</div>
        <div style="line-height:1.9;font-size:0.95em;padding:10px;
                    background:white;border-radius:8px;border:1px solid #e5e7eb;">
          {highlighted_text}
        </div>
        {keywords_html}
      </div>
    </div>"""


# =========================
# BUSINESS INDEX
# (loaded once from yelp_business.JSON)
# =========================
_business_index = {}   # business_id -> {name, city, state, stars, review_count}
_business_path  = ""


def _load_business_index(business_json_path):
    global _business_index, _business_path
    if _business_path == business_json_path and _business_index:
        return  # already loaded
    index = {}
    with open(business_json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = obj.get("business_id")
            if bid:
                index[bid] = {
                    "name":         obj.get("name", "Unknown"),
                    "city":         obj.get("city", ""),
                    "state":        obj.get("state", ""),
                    "stars":        obj.get("stars", 0),
                    "review_count": obj.get("review_count", 0),
                }
    _business_index = index
    _business_path  = business_json_path


def _build_all_business_choices(business_json_path):
    """Load all businesses from JSON and return sorted dropdown choices."""
    if not os.path.isfile(business_json_path):
        return []
    _load_business_index(business_json_path)
    choices = []
    for bid, info in _business_index.items():
        label = f"{info['name']}  [{bid}]"
        choices.append(label)
    choices.sort()
    return choices

# Pre-load a random sample of 500 businesses at startup
import random as _random
_DEFAULT_BUSINESS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "yelp_business.JSON")
print("Loading business index...")
_ALL_BUSINESS_CHOICES = _build_all_business_choices(_DEFAULT_BUSINESS_JSON)
_random.seed(42)
_ALL_BUSINESS_CHOICES = sorted(_random.sample(_ALL_BUSINESS_CHOICES, min(500, len(_ALL_BUSINESS_CHOICES))))
print(f"Sampled {len(_ALL_BUSINESS_CHOICES):,} businesses.")


def _extract_bid(dropdown_value):
    """Pull business_id out of the dropdown label string."""
    if not dropdown_value:
        return None
    m = re.search(r"\[([^\]]+)\]$", dropdown_value)
    return m.group(1) if m else None


# =========================
# REVIEW CACHE
# =========================
_reviews = []


def _load_business_reviews(review_json_path, business_id):
    """Load ALL reviews for a specific business from yelp_review.JSON."""
    reviews = []
    with open(review_json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("business_id") != business_id:
                continue
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            reviews.append({"text": text, "stars": obj.get("stars", 0)})
    return reviews


def _render(idx):
    if not _reviews:
        return "<p style='color:#6b7280'>Load a business first.</p>", idx, 0, gr.update()
    idx   = max(0, min(idx, len(_reviews) - 1))
    rev   = _reviews[idx]
    total = len(_reviews)
    label, prob_pos      = _run_cnn(rev["text"])
    tree_label, _        = _run_model(rev["text"], "Decision Tree")
    disagrees            = label != tree_label
    highlighted          = _highlight_review(rev["text"])
    top_pos, top_neg     = _top_keywords(rev["text"])
    card                 = _review_card_html(label, prob_pos, rev["stars"],
                                              highlighted, _keywords_html(top_pos, top_neg),
                                              disagrees)
    return card, idx, total, gr.update(value=idx + 1)


# =========================
# BUSINESS ANALYSIS
# =========================
def _make_charts(df, business_name, model_name):
    import pandas as pd
    POS, NEG, IND = "#22c55e", "#ef4444", "#4f46e5"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.patch.set_facecolor("#f9fafb")
    for ax in axes.flat:
        ax.set_facecolor("white")
    fig.suptitle(f"{business_name}  ·  {model_name}  ·  {len(df)} reviews",
                 fontsize=12, fontweight="bold", color="#111827", y=1.01)

    n_pos = (df["prediction"] == "POSITIVE").sum()
    n_neg = len(df) - n_pos

    # Donut
    ax = axes[0, 0]
    _, _, ats = ax.pie([n_pos, n_neg], labels=["Positive", "Negative"],
                       colors=[POS, NEG], autopct="%1.1f%%", startangle=90,
                       wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
                       textprops=dict(fontsize=10))
    for at in ats:
        at.set_fontsize(10); at.set_fontweight("bold"); at.set_color("white")
    ax.set_title("Positive vs Negative", fontsize=11, fontweight="bold", color="#374151", pad=10)

    # Star bar
    ax = axes[0, 1]
    x = np.arange(1, 6); w = 0.38
    ax.bar(x-w/2, [(df[df["stars"]==s]["prediction"]=="POSITIVE").sum() for s in range(1,6)],
           w, color=POS, label="Positive", edgecolor="white")
    ax.bar(x+w/2, [(df[df["stars"]==s]["prediction"]=="NEGATIVE").sum() for s in range(1,6)],
           w, color=NEG, label="Negative", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([f"★{s}" for s in range(1,6)], fontsize=10)
    ax.set_title("Predictions by Star Rating", fontsize=11, fontweight="bold", color="#374151", pad=10)
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

    # Score histogram
    ax = axes[1, 0]
    scores = df["score"].values
    bins   = np.linspace(-1, 1, 21)
    _, _, patches = ax.hist(scores, bins=bins, edgecolor="white")
    for patch, b in zip(patches, bins[:-1]):
        patch.set_facecolor(POS if b >= 0 else NEG)
    ax.axvline(scores.mean(), color=IND, linewidth=1.8, linestyle="--",
               label=f"Mean {scores.mean():+.2f}")
    ax.set_title("Score Distribution", fontsize=11, fontweight="bold", color="#374151", pad=10)
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

    # Avg confidence
    ax = axes[1, 1]
    cp = df.loc[df["prediction"]=="POSITIVE","confidence"].mean()
    cn = df.loc[df["prediction"]=="NEGATIVE","confidence"].mean()
    cp = cp if not np.isnan(cp) else 0
    cn = cn if not np.isnan(cn) else 0
    bars = ax.bar(["Positive","Negative"], [cp*100, cn*100],
                  color=[POS, NEG], edgecolor="white", width=0.45)
    for bar, v in zip(bars, [cp, cn]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{v*100:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#374151")
    ax.set_ylim(0, 110)
    ax.set_title("Avg Confidence by Prediction", fontsize=11, fontweight="bold", color="#374151", pad=10)
    ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    return fig


def analyze_business(dropdown_value, review_json_path, bulk_model):
    import pandas as pd
    global _reviews

    review_json_path = (review_json_path or "").strip()
    bid              = _extract_bid(dropdown_value)

    if not bid:
        err = "<p style='color:red'>Select a business from the dropdown first.</p>"
        return err, None, err, "—", 0, 0, gr.update(value=1, maximum=1)
    if not os.path.isfile(review_json_path):
        err = f"<p style='color:red'>Review JSON not found: <code>{review_json_path}</code></p>"
        return err, None, err, "—", 0, 0, gr.update(value=1, maximum=1)

    business_name = _business_index.get(bid, {}).get("name", bid)

    # ── Load all reviews for this business ────────────────────────────────────
    print(f"Loading reviews for: {business_name} ({bid})")
    all_reviews = _load_business_reviews(review_json_path, bid)

    if not all_reviews:
        msg = f"<p style='color:#f59e0b'>No reviews found for <strong>{business_name}</strong>.</p>"
        return msg, None, msg, "—", 0, 0, gr.update(value=1, maximum=1)

    # ── Bulk sentiment analysis ───────────────────────────────────────────────
    results = []
    for r in all_reviews:
        if r["stars"] == 3:
            continue  # skip ambiguous for bulk stats
        label, prob_pos = _run_model(r["text"], "CNN")
        results.append({"stars": r["stars"], "prediction": label,
                         "confidence": max(prob_pos, 1-prob_pos),
                         "score": sentiment_score(prob_pos)})

    if not results:
        msg = "<p style='color:#f59e0b'>All reviews were 3-star (ambiguous) — no sentiment data.</p>"
        return msg, None, msg, "—", 0, 0, gr.update(value=1, maximum=1)

    df    = pd.DataFrame(results)
    total = len(df)
    n_pos = (df["prediction"]=="POSITIVE").sum()
    n_neg = total - n_pos
    avg_s = df["score"].mean()
    avg_c = df["confidence"].mean()
    oc    = "#22c55e" if avg_s>=0.2 else "#ef4444" if avg_s<=-0.2 else "#f59e0b"
    ol    = "Doing Well" if avg_s>=0.2 else "Struggling" if avg_s<=-0.2 else "Mixed Reception"


    biz_info = _business_index.get(bid, {})
    info_str = f"{biz_info.get('city','')}, {biz_info.get('state','')}  ·  ★{biz_info.get('stars','')} on Yelp"

    summary_html = f"""
    <div style="font-family:sans-serif;padding:16px;background:#f9fafb;
                border-radius:12px;border:1px solid #e5e7eb;margin-bottom:8px;">
      <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
        <h2 style="margin:0;color:#111827;">{business_name}</h2>
        <span style="background:{oc};color:white;font-size:0.8em;padding:2px 12px;
                     border-radius:12px;font-weight:600;">{ol}</span>
      </div>
      <div style="color:#6b7280;font-size:0.85em;margin-bottom:14px;">{info_str}</div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;">
        {"".join(f'<div style="background:white;padding:12px;border-radius:8px;border:1px solid #e5e7eb;text-align:center;"><div style="font-size:1.5em;font-weight:700;color:{c};">{v}</div><div style="color:#6b7280;font-size:0.8em;">{l}</div></div>'
        for v,c,l in [
            (total,"#111827","Reviews Analysed"),
            (f"{n_pos/total*100:.1f}%","#22c55e",f"Positive ({n_pos})"),
            (f"{n_neg/total*100:.1f}%","#ef4444",f"Negative ({n_neg})"),
            (f"{avg_s:+.2f}",oc,"Avg Sentiment Score"),
            (f"{avg_c*100:.1f}%","#4f46e5","Avg Confidence"),
        ])}
      </div>
    </div>"""

    charts = _make_charts(df, business_name, bulk_model)

    # ── Load reviews into browser (all stars, not just non-3) ─────────────────
    _reviews = all_reviews
    total_r  = len(_reviews)
    card, idx, _, jump = _render(0)
    counter  = f"Review 1 of {total_r}"

    return (summary_html, charts, card, counter, idx, total_r,
            gr.update(value=1, minimum=1, maximum=total_r, label=f"Jump to (1–{total_r})"))


def _nav(idx, total, direction=0, jump=None):
    new_idx = (int(jump) - 1) if jump is not None else int(idx) + direction
    card, idx, total, jump_upd = _render(new_idx)
    return card, f"Review {idx+1} of {total}", idx, total, jump_upd


# =========================
# BUSINESS NAVIGATION
# =========================
def _biz_show(idx):
    """Return display label and index for a given business list position."""
    if not _ALL_BUSINESS_CHOICES:
        return "No businesses loaded", 0, len(_ALL_BUSINESS_CHOICES), gr.update(value=1)
    idx   = max(0, min(idx, len(_ALL_BUSINESS_CHOICES) - 1))
    total = len(_ALL_BUSINESS_CHOICES)
    label = re.sub(r'\s*\[[^\]]+\]$', '', _ALL_BUSINESS_CHOICES[idx])
    return label, idx, total


def biz_prev(idx, total):
    return _biz_show(int(idx) - 1)

def biz_next(idx, total):
    return _biz_show(int(idx) + 1)

def biz_jump(num, total):
    return _biz_show(int(num) - 1)


def analyze_selected(biz_idx, review_json_path, bulk_model):
    """Wrapper: pull the business choice by index and run analysis."""
    idx = max(0, min(int(biz_idx), len(_ALL_BUSINESS_CHOICES) - 1))
    return analyze_business(_ALL_BUSINESS_CHOICES[idx], review_json_path, bulk_model)


# =========================
# UI
# =========================
with gr.Blocks(theme=gr.themes.Default()) as demo:

    gr.HTML("""<style>
      button.lg { background:#4f46e5!important;color:white!important;
                  border:none!important;border-radius:10px!important;font-weight:600!important; }
      button.lg:hover { background:#3730a3!important; }
    </style>""")

    gr.Markdown("# Yelp Business Sentiment Analyzer")

    # ── Collapsible model metrics panel ──────────────────────────────────────
    with gr.Accordion("Model Metrics", open=False):
        gr.HTML(_load_metrics_html())

    # ── File paths (hidden, resolved automatically from data/ folder) ──────────
    _data_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    review_json  = gr.State(value=os.path.join(_data_dir, "yelp_review.JSON"))
    business_json = gr.State(value=os.path.join(_data_dir, "yelp_business.JSON"))

    bulk_model = gr.State(value="CNN")

    # ── Business navigator ───────────────────────────────────────────────────
    gr.Markdown("### Select a Business")
    biz_current_idx = gr.State(value=0)
    biz_total       = gr.State(value=len(_ALL_BUSINESS_CHOICES))

    with gr.Row():
        biz_prev_btn    = gr.Button("◀  Previous", variant="primary", scale=1)
        biz_counter_out = gr.Textbox(
            value=_ALL_BUSINESS_CHOICES[0] if _ALL_BUSINESS_CHOICES else "No businesses loaded",
            show_label=False, interactive=False, scale=4
        )
        biz_next_btn    = gr.Button("Next  ▶", variant="primary", scale=1)

    analyze_btn = gr.Button("Analyze Selected Business", variant="primary")

    # ── Business report ───────────────────────────────────────────────────────
    gr.Markdown("## Business Report")
    biz_summary = gr.HTML("<p style='color:#9ca3af'>Search for a business and click Analyze.</p>")
    biz_charts  = gr.State()  # charts disabled

    gr.Markdown("---")

    # ── Review browser ────────────────────────────────────────────────────────
    gr.Markdown("## Review Browser")
    gr.Markdown("CNN classifies each review · Naive Bayes highlights key sentiment words")

    current_idx = gr.State(value=0)
    total_count = gr.State(value=0)

    review_card = gr.HTML()

    with gr.Row():
        prev_btn    = gr.Button("◀  Previous", variant="primary", scale=1)
        counter_out = gr.Textbox(value="Select and analyze a business to begin",
                                  show_label=False, interactive=False, scale=2)
        next_btn    = gr.Button("Next  ▶", variant="primary", scale=1)

    jump_num = gr.State(value=1)
    nav_outs = [review_card, counter_out, current_idx, total_count, jump_num]

    # Wire up business navigation
    biz_nav_outs = [biz_counter_out, biz_current_idx, biz_total]

    biz_prev_btn.click(fn=biz_prev, inputs=[biz_current_idx, biz_total], outputs=biz_nav_outs)
    biz_next_btn.click(fn=biz_next, inputs=[biz_current_idx, biz_total], outputs=biz_nav_outs)
    # Wire up analyze
    analyze_btn.click(
        fn=analyze_selected,
        inputs=[biz_current_idx, review_json, bulk_model],
        outputs=[biz_summary, biz_charts, review_card, counter_out,
                 current_idx, total_count, jump_num]
    )

    # Navigation
    prev_btn.click(fn=lambda i,t: _nav(i,t,direction=-1),
                   inputs=[current_idx, total_count], outputs=nav_outs)
    next_btn.click(fn=lambda i,t: _nav(i,t,direction=1),
                   inputs=[current_idx, total_count], outputs=nav_outs)


demo.launch()