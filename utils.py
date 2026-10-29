import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import AutoTokenizer,  AutoModelForSequenceClassification

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


def init_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "True", 1:"Fake"},
        label2id={"True": 0, "Fake": 0}
    )
    return model

def tokenize_dataset(x, y, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = Dataset.from_dict({
        "text": x.fillna("").astype(str).tolist(),
        "labels": y.astype(int).tolist()
    })

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256
        )

    return dataset.map(
        tokenize,
        batched=True
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }