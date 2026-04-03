import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def encode_data(df):
    df = pd.get_dummies(df, drop_first=True)
    return df


def preprocess_data(path):
    df = load_data(path)
    df = clean_data(df)
    df = encode_data(df)
    return df