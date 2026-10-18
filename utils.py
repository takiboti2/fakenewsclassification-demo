import pandas as pd
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42

def process_data():
    true = pd.read_csv("data/True.csv")

    fake = pd.read_csv("data/Fake.csv")

    true["fake"] = 0

    fake["fake"] = 1 

    #boolean column for fake, 1 is fake, 0 is true

    df_raw = pd.concat([true, fake])

    df_raw = (df_raw.sample(frac=1,
                            random_state=42)).reset_index(drop=True)

    df_raw["full_text"] = df_raw["title"] + " " + df_raw["text"]

    cleaned = df_raw[["full_text", "fake"]]

    return cleaned

def split_data(df):
    #returns as X_train, X_val, X_test, y_train, y_val, y_test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify= df["fake"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify = temp_df["fake"]
    )

    X_train, y_train = train_df["full_text"], train_df["fake"]

    X_val, y_val = val_df["full_text"], val_df["fake"]

    X_test, y_test = test_df["full_text"], test_df["fake"]

    return X_train, X_val, X_test, y_train, y_val, y_test
