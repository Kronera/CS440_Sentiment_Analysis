# ================================================
#  preprocessing/cleaner.py — Text cleaning
# ================================================

import re
import pandas as pd

# Preprocessing steps:
# 1. Remove HTML tags
# 2. Remove non-alphabetic characters
# 3. Convert to lowercase
def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)       # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep only letters
    text = text.lower().strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing")

    df = df.copy()
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    return df
