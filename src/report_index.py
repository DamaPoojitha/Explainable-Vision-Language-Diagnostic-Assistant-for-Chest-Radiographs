"""
Builds a retrieval index over real OpenI radiology reports so the report
generator can produce clinically-worded text instead of a rigid template.

NOTE: OpenI's downloadable format varies (XML per-report, or a flattened
CSV export). `load_openi_reports` below assumes a CSV with columns
["report_id", "findings", "impression", "labels"] -- adjust the loader to
match whatever format you actually download. The rest of the pipeline
doesn't care about the source format as long as this function returns
a DataFrame with those columns.
"""

import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from dataset import CONDITIONS


def load_openi_reports(csv_path):
    df = pd.read_csv(csv_path)
    required = {"findings", "impression", "labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"OpenI CSV is missing columns {missing}. Adjust load_openi_reports() "
            f"to match your actual export format."
        )
    df["text"] = (df["findings"].fillna("") + " " + df["impression"].fillna("")).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def build_index(openi_csv, out_path="models/report_index.pkl"):
    df = load_openi_reports(openi_csv)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix, "reports": df},
            f,
        )
    print(f"Built index over {len(df)} reports -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/openi/reports.csv")
    parser.add_argument("--out", default="models/report_index.pkl")
    args = parser.parse_args()
    build_index(args.csv, args.out)