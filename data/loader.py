# ================================================
#  data/loader.py — Load and split the dataset
# ================================================

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, TEST_SIZE, RANDOM_STATE

# Loading data from csv
def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

# Train/test split with stratification
def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    print(f"Train samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}")
    print()

    return X_train, X_test, y_train, y_test
