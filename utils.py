import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
from lingua import Language, LanguageDetectorBuilder
RANDOM_SEED = 42

def load_data():
    df_raw = pd.read_csv("data/WELFake_Dataset.csv")

    df_raw = df_raw.drop(columns=["Unnamed: 0"])
    df_raw = df_raw.rename({"label": "labels"}, axis=1)

    df_raw = df_raw.drop_duplicates()
    df_raw["full_text"] = df_raw["title"].fillna("").str.strip() + " " + df_raw["text"].fillna("").str.strip()
    return df_raw
def is_english(text):
    if not isinstance(text, str) or not text.strip():
        return False

    detector = (
    LanguageDetectorBuilder
    .from_all_languages()
    .build()
)

    language = detector.detect_language_of(text[:2000])

    return language == Language.ENGLISH

def process_data():
    df =load_data()
    df["full_text"] = df["title"].fillna("").str.strip() + " " + df["text"].fillna("").str.strip()

    df["full_text"] = df["full_text"].str.replace(
    r"^[A-Z][A-Z\s,.'/-]{0,80}\s+\(Reuters\)\s*[-–—]\s*",
    "",
    regex=True
    )
    #remove reuters source tags
    
    english_mask = df["title"].apply(is_english)

    df = df[english_mask]
    


    cleaned = df[["title","labels"]]
    cleaned = (cleaned.sample(frac=1,
                    random_state=RANDOM_SEED)
                    .reset_index(drop=True))
    return cleaned

def split_data():
    df = process_data()
    #returns as X_train, X_val, X_test, y_train, y_val, y_test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify= df["labels"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify = temp_df["labels"]
    )

    train_df.to_csv("data/train.csv", index=False)

    val_df.to_csv("data/val.csv", index=False)

    test_df.to_csv("data/test.csv", index=False)

    X_train, y_train = train_df["title"], train_df["labels"]

    X_val, y_val = val_df["title"], val_df["labels"]

    X_test, y_test = test_df["title"], test_df["labels"]

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_splits():

    counter = 0
    paths = {}
    splits = ["train", "val", "test"]
    for split in splits:
        paths[split] = Path(f"data/{split}.csv")

        if paths[split].is_file():
            counter += 1
            print(counter)

    if counter != 3:
        return split_data()

    dfs = {}
    Xs = {}
    ys = {}
    for split in splits:
        dfs[split] = pd.read_csv(paths[split])
        Xs[split], ys[split] = dfs[split]["title"], dfs[split]["labels"]

    return (*Xs.values(), *ys.values())