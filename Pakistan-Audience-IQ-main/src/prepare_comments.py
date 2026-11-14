import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"


def clean_text(value):
    text = unicodedata.normalize("NFC", value)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def prepare_comments(comments, posts):
    required = {"comment_id", "post_id", "comment_text", "language_type", "data_origin"}
    missing = required - set(comments.columns)
    if missing:
        raise ValueError(f"Missing comment columns: {sorted(missing)}")

    if not {"post_id", "comments_count"}.issubset(posts.columns):
        raise ValueError("Posts require post_id and comments_count columns.")

    comments = comments.copy()
    posts = posts.copy()

    for column in ["comment_id", "post_id"]:
        comments[column] = comments[column].str.strip()
        if comments[column].eq("").any():
            raise ValueError(f"Blank {column} values found.")

    posts["post_id"] = posts["post_id"].str.strip()
    if posts["post_id"].eq("").any() or posts["post_id"].duplicated().any():
        raise ValueError("Post IDs must be nonblank and unique.")
    if comments["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs found.")
    if not comments["post_id"].isin(posts["post_id"]).all():
        raise ValueError("Some comments reference missing posts.")

    actual_counts = comments.groupby("post_id").size()
    expected_counts = pd.to_numeric(posts["comments_count"], errors="raise")
    actual = posts["post_id"].map(actual_counts).fillna(0)
    if not actual.eq(expected_counts).all():
        raise ValueError("Post comment counts do not match the comment table.")

    comments["clean_text"] = comments["comment_text"].map(clean_text)
    blank = comments["clean_text"].eq("")
    comments["repeated_text"] = comments.duplicated("clean_text", keep=False)
    comments["text_length"] = comments["clean_text"].str.len()

    report = {
        "data_origin": "synthetic",
        "input_comments": len(comments),
        "blank_after_cleaning": int(blank.sum()),
        "clustering_ready_comments": int((~blank).sum()),
        "unique_clean_texts": int(comments.loc[~blank, "clean_text"].nunique()),
        "records_with_repeated_text": int(comments["repeated_text"].sum()),
        "repeated_text_policy": "Retained and flagged",
        "language_labels": "Provided by the synthetic generator",
        "languages": {
            str(label): int(count)
            for label, count in comments["language_type"].value_counts().items()
        },
    }

    # Keep blank records in the source CSV; exclude them from clustering input.
    return comments.loc[~blank].copy(), report


def main():
    try:
        comments = pd.read_csv(
            DATA / "comments.csv", encoding="utf-8-sig",
            dtype=str, keep_default_na=False,
        )
        posts = pd.read_csv(
            DATA / "posts.csv", encoding="utf-8-sig",
            dtype=str, keep_default_na=False,
        )
        prepared, report = prepare_comments(comments, posts)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Preparation failed: {error}") from None

    output = DATA / "prepared_comments.csv"
    prepared.to_csv(output, index=False, encoding="utf-8-sig")

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "preparation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("COMMENT PREPARATION COMPLETE")
    print(f"Input comments: {report['input_comments']:,}")
    print(f"Ready for clustering: {report['clustering_ready_comments']:,}")
    print(f"Unique cleaned texts: {report['unique_clean_texts']:,}")
    print(f"Blank after cleaning: {report['blank_after_cleaning']:,}")
    print("PASS: Unique IDs, post links, and comment counts verified.")
    print(f"Prepared CSV: {output}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
