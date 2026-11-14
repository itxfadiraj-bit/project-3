import json
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


def main():
    source = DATA / "prepared_comments.csv"
    if not source.is_file():
        raise SystemExit("Run src/prepare_comments.py first.")

    comments = pd.read_csv(
        source, encoding="utf-8-sig", dtype=str, keep_default_na=False
    )
    required = {"comment_id", "clean_text", "language_type"}
    if not required.issubset(comments.columns):
        raise SystemExit("Prepared comments are missing required columns.")

    if comments["clean_text"].str.strip().eq("").any():
        raise SystemExit("Blank cleaned text found. Run preparation again.")

    unique = comments.drop_duplicates("clean_text").reset_index(drop=True)
    if len(unique) < 4:
        raise SystemExit("At least four unique comments are needed.")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5_000,
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(unique["clean_text"])

    candidates = []
    best_model = None
    best_score = -float("inf")

    for clusters in range(2, min(10, len(unique) - 1) + 1):
        model = KMeans(n_clusters=clusters, n_init=10, random_state=SEED)
        labels = model.fit_predict(features)
        label_count = len(np.unique(labels))
        if not 1 < label_count < len(unique):
            continue

        score = float(silhouette_score(features, labels, metric="cosine"))
        candidates.append({
            "clusters": clusters,
            "cosine_silhouette": score,
        })
        print(f"Candidate k={clusters}: silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise SystemExit("No valid clustering solution found.")

    unique["cluster_id"] = best_model.labels_
    label_lookup = unique.set_index("clean_text")["cluster_id"]
    comments["cluster_id"] = comments["clean_text"].map(label_lookup).astype(int)

    terms = vectorizer.get_feature_names_out()
    summaries = []

    for cluster_id in sorted(unique["cluster_id"].unique()):
        positions = np.flatnonzero(best_model.labels_ == cluster_id)
        centroid = best_model.cluster_centers_[cluster_id]

        # TF-IDF rows have unit length, so dot products rank centroid similarity.
        similarities = np.asarray(features[positions].dot(centroid)).ravel()
        representative_positions = positions[
            np.argsort(-similarities, kind="stable")[:3]
        ]
        examples = unique.iloc[representative_positions]["clean_text"].tolist()
        top_positions = np.argsort(-centroid, kind="stable")[:8]
        top_terms = [str(terms[index]) for index in top_positions if centroid[index] > 0]

        members = comments.loc[comments["cluster_id"].eq(cluster_id)]
        summaries.append({
            "cluster_id": int(cluster_id),
            "comments": len(members),
            "unique_texts": len(positions),
            "share_pct": round(100 * len(members) / len(comments), 2),
            "top_terms": ", ".join(top_terms),
            "example_1": examples[0],
            "example_2": examples[1] if len(examples) > 1 else "",
            "example_3": examples[2] if len(examples) > 2 else "",
        })

    summary = pd.DataFrame(summaries)
    language_counts = (
        comments.groupby(["cluster_id", "language_type"])
        .size().rename("comments").reset_index()
    )
    REPORTS.mkdir(parents=True, exist_ok=True)

    comments.to_csv(DATA / "clustered_comments.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORTS / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    language_counts.to_csv(
        REPORTS / "cluster_languages.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(candidates).to_csv(REPORTS / "cluster_candidates.csv", index=False)

    report = {
        "data_origin": "synthetic",
        "comments_assigned": len(comments),
        "unique_texts_used_for_training": len(unique),
        "selected_clusters": best_model.n_clusters,
        "selection": "Highest cosine silhouette among k=2 through min(10, unique texts - 1)",
        "cosine_silhouette": best_score,
        "seed": SEED,
        "review_notes": [
            "Silhouette measures text separation, not marketing accuracy.",
            "TF-IDF may separate languages rather than audience needs.",
            "Review cluster terms, examples, and language composition before naming groups.",
            "Synthetic template data does not establish real customer behaviour.",
        ],
    }
    (REPORTS / "clustering_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if summary["comments"].sum() != len(comments):
        raise RuntimeError("Cluster counts do not match the input record count.")

    print("\nCLUSTERING COMPLETE")
    print(f"Comments assigned: {len(comments):,}")
    print(f"Unique texts used for training: {len(unique):,}")
    print(f"Selected clusters: {best_model.n_clusters}")
    print(f"Cosine silhouette: {best_score:.4f}")
    print("\nCluster summary:")
    print(summary[["cluster_id", "comments", "unique_texts", "top_terms"]].to_string(index=False))
    print("\nPASS: All input comments received a cluster.")


if __name__ == "__main__":
    main()
