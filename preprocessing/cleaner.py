import re
import pandas as pd

_nlp = None
_STOP_WORDS = None

# Load spaCy model and cache it
def _get_nlp():
    global _nlp, _STOP_WORDS
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        _STOP_WORDS = _nlp.Defaults.stop_words - {
            "not", "no", "nor", "never", "nobody", "nothing",
            "neither", "nowhere", "hardly", "barely", "scarcely",
        }
    return _nlp

# Negations stored for the NLP
_NEGATION_RE = re.compile(
    r"\b(not|no|never|nor|nobody|nothing|neither|nowhere|hardly|barely|scarcely)\b"
    r"((?:\s+\w+){1,4})",
    re.IGNORECASE,)

# Find and flag negation words with the _NEG suffix
def _apply_negation(text: str) -> str:
    def _tag(match):
        trigger = match.group(1)
        window  = match.group(2)
        tagged  = re.sub(r"(\w+)", r"\1_NEG", window)
        return trigger + tagged
    return _NEGATION_RE.sub(_tag, text)

_KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# Cleaning review text for use inside the models
def clean_text(text: str) -> str:
    import contractions

    nlp = _get_nlp()
    text = contractions.fix(text)
    text = re.sub(r"<.*?>", " ", text)
    text = _apply_negation(text)
    text = re.sub(r"[^a-zA-Z\s_]", " ", text)
    
    tokens = []
    
    for token in nlp(text):
        word = token.text

        if word.endswith("_NEG"):
            root = word[:-4]
            doc  = nlp(root)
            lemma = doc[0].lemma_ if doc else root
            tokens.append(lemma.lower() + "_NEG")
        else:
            if token.pos_ in _KEEP_POS and token.lemma_.lower() not in _STOP_WORDS:
                tokens.append(token.lemma_.lower())

    return " ".join(tokens).strip()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("2-3 Minutes")
    df = df.copy()
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    return df
