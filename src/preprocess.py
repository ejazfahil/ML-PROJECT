"""Preprocessing. 2024-03-06"""
import pandas as pd

FEATS = ["age","workclass","education","occupation","race","sex","hours-per-week"]

def load(path):
    df = pd.read_csv(path, na_values=[" ?"])
    df.dropna(inplace=True)
    df["income"] = (df["income"].str.strip()==">50K").astype(int)
    return df

def encode(df):
    return pd.get_dummies(df, columns=df.select_dtypes("object").columns, drop_first=True)
