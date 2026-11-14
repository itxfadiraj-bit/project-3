import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
REPORTS = ROOT / "reports"
SEED = 42

# Inspectable vocabulary for this demonstration category.
TRANSLATIONS = {
    "qeemat": "price", "قیمت": "price",
    "kapra": "fabric", "کپڑا": "fabric",
    "garmi": "summer", "گرمیوں": "summer",
    "acha": "good", "اچھا": "good",
    "soft": "soft",
    "سائز": "size",
    "ڈیلیوری": "delivery", "parcel": "delivery", "پارسل": "delivery",
    "din": "days", "دن": "days",
    "haftay": "week", "ہفتے": "week",
    "pichlay": "last", "پچھلے": "last",
    "waiting": "waiting",
    "nahi": "not", "نہیں": "not",
    "aya": "arrived", "آیا": "arrived",
    "رعایت": "discount", "sale": "discount",
    "mil": "available", "دستیاب": "available",
    "branch": "store", "برانچ": "store",
    "lena": "buy", "خریدنا": "buy",
    "آرڈر": "order", "ordered": "order",
    "colours": "colour", "colors": "colour", "رنگ": "colour",
    "pyaray": "beautiful", "خوبصورت": "beautiful",
    "zyada": "high", "زیادہ": "high",
}

STOPWORDS = set("""
a an the this that of for in on at to is are do does have has any
what how many will take you your i want am and still it
ye is ki ka ke ko hai hain kya mein par ho hogi kitne kaisay
bohat sakta abhi tak tha karun
اس یہ کی کا کے کو ہے ہیں کیا میں پر کتنے ہوگی کوئی
بہت ابھی تک تھا چاہتی ہوں کیسے کروں
lawn suit suits printed embroidered سوٹ
lahore karachi islamabad multan faisalabad لاہور کراچی
""".split())


def normalize_features(text):
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    mapped = [TRANSLATIONS.get(token, token) for token in tokens]
    useful = [token for token in mapped if token not in STOPWORDS]
    # Keep usable text even when the dictionary removes all its tokens.
    return " ".join(useful) or text


def main():
    comments = pd.read_csv(
        DATA / "prepared_comments.csv",
        encoding="utf-8-sig", dtype=str, keep_default_na=False,
    )
    comments["feature_text"] = comments["clean_text"].map(normalize_features)

    # Fit on unique feature texts and retain all comments in the final results.
    unique = comments.drop_duplicates("feature_text").reset_index(drop=True)
    if len(unique) < 4:
        raise SystemExit("Too few unique feature texts for clustering.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5_000)
    features = vectorizer.fit_transform(unique["feature_text"])
    candidates = []
    best_model = None
    best_score = -float("inf")

    for clusters in range(2, min(10, len(unique) - 1) + 1):
        model = KMeans(n_clusters=clusters, n_init=10, random_state=SEED)
        labels = model.fit_predict(features)
        if not 1 < len(np.unique(labels)) < len(unique):
            continue
        score = float(silhouette_score(features, labels, metric="cosine"))
        candidates.append({"clusters": clusters, "cosine_silhouette": score})
        print(f"Candidate k={clusters}: silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise SystemExit("No valid clustering solution found.")

    unique["cluster_id"] = best_model.labels_
    lookup = unique.set_index("feature_text")["cluster_id"]
    comments["cluster_id"] = comments["feature_text"].map(lookup).astype(int)
    terms = vectorizer.get_feature_names_out()
    summary = []

    for cluster_id in sorted(comments["cluster_id"].unique()):
        members = comments.loc[comments["cluster_id"].eq(cluster_id)]
        centroid = best_model.cluster_centers_[cluster_id]
        top = np.argsort(-centroid, kind="stable")[:8]
        top_terms = [str(terms[index]) for index in top if centroid[index] > 0]
        examples = members.drop_duplicates("clean_text")["comment_text"].head(3)
        summary.append({
            "cluster_id": int(cluster_id),
            "comments": len(members),
            "unique_original_texts": members["clean_text"].nunique(),
            "languages_present": members["language_type"].nunique(),
            "top_terms": ", ".join(top_terms),
            "examples": " | ".join(examples),
        })

    summary = pd.DataFrame(summary)
    if summary["comments"].sum() != len(comments):
        raise RuntimeError("Cluster counts do not match the input.")

    REPORTS.mkdir(parents=True, exist_ok=True)
    comments.to_csv(
        DATA / "refined_clustered_comments.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        REPORTS / "refined_cluster_summary.csv", index=False, encoding="utf-8-sig"
    )
    comments.groupby(["cluster_id", "language_type"]).size().rename(
        "comments"
    ).reset_index().to_csv(
        REPORTS / "refined_cluster_languages.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(candidates).to_csv(
        REPORTS / "refined_cluster_candidates.csv", index=False
    )

    report = {
        "data_origin": "synthetic",
        "method": "Dictionary-assisted TF-IDF and KMeans",
        "comments_assigned": len(comments),
        "unique_feature_texts": len(unique),
        "selected_clusters": best_model.n_clusters,
        "cosine_silhouette": best_score,
        "seed": SEED,
        "review_notes": [
            "Vocabulary was designed for the demonstration category.",
            "This vocabulary is not a general translation system.",
            "Baseline and refined scores use different feature spaces.",
            "Review examples and language composition to assess usefulness.",
        ],
    }
    (REPORTS / "refined_clustering_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\nREFINED CLUSTERING COMPLETE")
    print(f"Comments assigned: {len(comments):,}")
    print(f"Unique feature texts: {len(unique):,}")
    print(f"Selected clusters: {best_model.n_clusters}")
    print(f"Cosine silhouette: {best_score:.4f}")
    print(summary.drop(columns="examples").to_string(index=False))
    print("\nPASS: All comments received a cluster.")


if __name__ == "__main__":
    main()
