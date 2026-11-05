from pathlib import Path
import re
import unicodedata
import warnings

from datasets import Dataset
from lingua import Language, LanguageDetectorBuilder
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


RANDOM_SEED = 42

def load_data():
    #simple loading utility function, used for later processing pipeline steps and eda
    #does limited cleaning and preprocessing, makes a full_text column, with na handling
    #removes trailing whitespace from title and text, and drops duplicates
    #but keeps missing values for data quality checks
    df_raw = pd.read_csv("data/WELFake_Dataset.csv")

    df_raw = df_raw.drop(columns=["Unnamed: 0"])
    df_raw = df_raw.rename({"label": "labels"}, axis=1)
    #limited cleaning and preprocessing. remove trailing whitespace and drop duplicates
    #keep missing values at this step
    df_raw["title"] = df_raw["title"].str.strip() 
    df_raw["text"] = df_raw["text"].str.strip()
    #make a full text column, fillna with empty strings so rows with either missing title or text dont get coalesced to na
    df_raw["full_text"] = df_raw["title"].fillna("") + " " + df_raw["text"].fillna("")
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

# Restrict removal to complete bracketed markers, preserving ordinary sentence words.
MEDIA_TOKEN = r"(?:details?|images?|audio|tweets?|(?:live[\s–—-]*)?videos?|screenshots?|watch)"
MARKER_CONTENT = rf"\+?\s*{MEDIA_TOKEN}(?:\s*[/+&,]\s*{MEDIA_TOKEN})*"
HEADLINE_MARKER = re.compile(
    rf"\(\s*{MARKER_CONTENT}\s*\)|\[\s*{MARKER_CONTENT}\s*\]|\{{\s*{MARKER_CONTENT}\s*\}}",
    flags=re.IGNORECASE,
)

# Only strip these outlet names in trailing attribution positions.
PUBLISHER_SUFFIXES = (
    "TruthFeed", "New Eastern Outlook", "EndingFed News Network", "Daily Wire",
    "RedFlag News", "The Vineyard of the Saker", "Katehon Think Tank. Geopolitics & Tradition",
    "Conservative Daily Post", "Collective Evolution", "Russia & India Report",
    "New Century Times", "American Lookout", "Gomerblog", "MAGAFeed",
    "The Federalist Papers", "Addicting Info | The Knowledge You Crave",
    "Twitchy.com", "ClickHole", "RealClearPolitics", "OffGuardian", "Top Right News",
)
PUBLISHER_SUFFIX = re.compile(
    r"\s+(?:[|–—-])\s*(?:" + "|".join(
        r"\s+".join(re.escape(word) for word in name.split())
        for name in sorted(PUBLISHER_SUFFIXES, key=len, reverse=True)
    ) + r")\s*$",
    flags=re.IGNORECASE,
)
URL_ONLY = re.compile(r"(?:(?:https?://|www\.)\S+\s*)+", flags=re.IGNORECASE)



# Explicit outlet names only: avoid ambiguous standalone words such as AP or Time.
SOURCE_IDENTIFIERS = (
    "Reuters", "Thomson Reuters", "The New York Times", "New York Times",
    "NYTimes", "nytimes.com", "InfoWars", "Info Wars", "infowars.com",
    "Associated Press", "The Associated Press", "The Washington Post",
    "Washington Post", "The Guardian", "BBC News", "CNN", "Fox News",
    "ABC News", "CBS News", "NBC News", "MSNBC", "Huffington Post",
    "HuffPost", "Breitbart", "BuzzFeed", "BuzzFeed News", "21st Century Wire",
    "21Wire", "Daily Mail", "The Daily Mail", "USA Today", "Politico",
)
SOURCE_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(
        r"\s+".join(re.escape(word) for word in name.split())
        for name in sorted(SOURCE_IDENTIFIERS, key=len, reverse=True)
    ) + r")(?!\w)",
    flags=re.IGNORECASE,
)


REMOVED_WORDS = re.compile(r"(?<!\w)factbox(?!\w)", flags=re.IGNORECASE)


def clean_title(text):
    """Remove media markers and configured outlet names, including in sentences."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.strip()
    if URL_ONLY.fullmatch(text):
        return ""
    text = HEADLINE_MARKER.sub(" ", text)
    # Repeat to handle multiple known attribution suffixes, never arbitrary dash text.
    while True:
        cleaned = PUBLISHER_SUFFIX.sub("", text.rstrip())
        if cleaned == text:
            break
        text = cleaned
    text = SOURCE_PATTERN.sub(" ", text)
    text = REMOVED_WORDS.sub(" ", text)
    # Remove wrappers emptied by source/marker removal and dangling attribution separators.
    text = re.sub(r"\(\s*\)|\[\s*\]|\{\s*\}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" |:;,-–—")


def normalize_title(text):
    """Conservative identity key: Unicode, case, and whitespace normalization."""
    return clean_title(text).casefold()


def deduplicate_titles(df):
    df = df[["title", "labels"]].dropna().copy()
    df["title"] = df["title"].map(clean_title)
    df["headline_key"] = df["title"].map(normalize_title)
    df = df[df["headline_key"].ne("")]
    conflicts = df.groupby("headline_key")["labels"].nunique()
    conflicts = conflicts[conflicts > 1].index
    if len(conflicts):
        warnings.warn(f"Excluding {len(conflicts)} headlines with conflicting labels.")
    return (df[~df["headline_key"].isin(conflicts)]
            .drop_duplicates("headline_key").drop(columns="headline_key")
            .reset_index(drop=True))


def validate_splits(dfs):
    seen = set()
    for name, df in dfs.items():
        if df[["title", "labels"]].isna().any().any() or not df["labels"].isin([0, 1]).all():
            raise ValueError(f"Invalid titles or labels in {name} split")
        keys = df["title"].map(normalize_title)
        if keys.eq("").any() or keys.duplicated().any() or seen.intersection(keys):
            raise ValueError("Duplicate headlines in cached splits; regenerate splits and retrain models.")
        seen.update(keys)


def process_data():
    df =load_data()

    #remove reuters source tags
    
    df["title"] = df["title"].map(lambda text: clean_title(text) if isinstance(text, str) else text)
    english_mask = df["title"].apply(is_english)

    df = df[english_mask]
    


    cleaned = deduplicate_titles(df)
    cleaned = (cleaned.sample(frac=1,
                    random_state=RANDOM_SEED)
                    .reset_index(drop=True))
    return cleaned

def split_data(df=None):
    df = process_data() if df is None else deduplicate_titles(df)
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

    validate_splits({"train": train_df, "val": val_df, "test": test_df})
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

    if counter != 3:
        return split_data()

    dfs = {}
    Xs = {}
    ys = {}
    for split in splits:
        dfs[split] = pd.read_csv(paths[split])
        dfs[split]["title"] = dfs[split]["title"].map(clean_title)
        Xs[split], ys[split] = dfs[split]["title"], dfs[split]["labels"]

    validate_splits(dfs)
    return (*Xs.values(), *ys.values())


def init_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "True", 1:"Fake"},
        label2id={"True": 0, "Fake": 1}
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

    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(shifted_logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return classification_metrics(labels, predictions, probabilities)


def classification_metrics(labels, predictions, probabilities=None):
    """Binary classification scores; optional probability columns must be ordered [0, 1].

    Log loss is returned separately from Trainer's own cross-entropy loss.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }
    if probabilities is not None:
        metrics["log_loss"] = log_loss(labels, probabilities, labels=[0, 1])
    return metrics
