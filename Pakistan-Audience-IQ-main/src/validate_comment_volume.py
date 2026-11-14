import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_campaigns import MIN_PEERS, THRESHOLD, robust_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
REPORTS = ROOT / "reports"


def main():
    rng = np.random.default_rng(42)
    posts = pd.read_csv(DATA / "posts.csv", encoding="utf-8-sig")
    source_comments = pd.read_csv(
        DATA / "comments.csv", encoding="utf-8-sig",
        dtype=str, keep_default_na=False,
    )

    if posts["post_id"].duplicated().any():
        raise ValueError("Post IDs must be unique.")

    posts["age_band"] = pd.cut(
        posts["days_since_post"],
        bins=[-1, 7, 30, np.inf],
        labels=["0-7 days", "8-30 days", "31+ days"],
    ).astype(str)
    peers = ["platform", "content_type", "age_band"]
    posts["peer_posts"] = posts.groupby(
        peers, observed=True
    )["post_id"].transform("size")

    eligible_positions = np.flatnonzero(posts["peer_posts"].ge(MIN_PEERS))
    if len(eligible_positions) < 12:
        raise ValueError("Need at least 12 posts in sufficiently large peer groups.")

    planted_positions = rng.choice(eligible_positions, size=12, replace=False)
    planted_ids = set(posts.iloc[planted_positions]["post_id"])

    # Generate background volumes independently of the detector.
    posts["comments_count"] = rng.integers(2, 9, size=len(posts))
    posts.loc[posts["post_id"].isin(planted_ids), "comments_count"] = 200

    # Build actual linked comments so counts agree with the posts table.
    pool = source_comments.drop_duplicates("comment_text").reset_index(drop=True)
    if pool.empty:
        raise ValueError("The source comment table is empty.")

    comments = []
    for post in posts.itertuples(index=False):
        selections = rng.integers(0, len(pool), size=int(post.comments_count))
        for index in selections:
            original = pool.iloc[index]
            comments.append({
                "comment_id": f"V{len(comments) + 1:07d}",
                "post_id": post.post_id,
                "comment_text": original["comment_text"],
                "language_type": original["language_type"],
                "data_origin": "synthetic",
            })
    comments = pd.DataFrame(comments)

    actual_counts = comments.groupby("post_id").size()
    if not posts["post_id"].map(actual_counts).eq(posts["comments_count"]).all():
        raise ValueError("Generated comments do not match post counts.")

    posts["log_comments"] = np.log1p(posts["comments_count"])
    posts["comment_volume_score"] = posts.groupby(
        peers, observed=True
    )["log_comments"].transform(robust_score)
    posts["review_eligible"] = (
        posts["peer_posts"].ge(MIN_PEERS)
        & posts["comment_volume_score"].notna()
    )
    posts["unusual_comment_volume"] = (
        posts["review_eligible"]
        & posts["comment_volume_score"].abs().gt(THRESHOLD)
    )

    detected_ids = set(
        posts.loc[posts["unusual_comment_volume"], "post_id"]
    )
    background = posts.loc[~posts["post_id"].isin(planted_ids)]
    background_eligible = int(background["review_eligible"].sum())
    background_flags = len(detected_ids - planted_ids)

    evaluation = {
        "data_origin": "synthetic controlled volume test",
        "seed": 42,
        "posts": len(posts),
        "comments": len(comments),
        "background_comments_per_post": "2 through 8",
        "planted_comments_per_post": 200,
        "planted_posts": len(planted_ids),
        "planted_posts_detected": len(planted_ids & detected_ids),
        "planted_posts_missed": len(planted_ids - detected_ids),
        "eligible_background_posts": background_eligible,
        "background_posts_flagged": background_flags,
        "background_flag_rate": (
            background_flags / background_eligible if background_eligible else None
        ),
        "notes": [
            "Planted IDs are used only for evaluation.",
            "This evaluates an obvious high-volume simulation.",
            "Results do not establish detection accuracy on real social media.",
        ],
    }

    demo = DATA / "volume_demo"
    demo.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    posts.drop(columns="log_comments").to_csv(
        demo / "posts.csv", index=False, encoding="utf-8-sig"
    )
    comments.to_csv(demo / "comments.csv", index=False, encoding="utf-8-sig")

    # Save the answer key separately from the scored posts.
    pd.DataFrame({"post_id": sorted(planted_ids)}).to_csv(
        REPORTS / "volume_test_planted_posts.csv", index=False
    )
    posts.loc[posts["unusual_comment_volume"]].drop(
        columns="log_comments"
    ).to_csv(REPORTS / "unusual_comment_volume.csv", index=False)

    (REPORTS / "volume_validation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )

    print("COMMENT-VOLUME VALIDATION COMPLETE")
    print(f"Posts: {len(posts):,}")
    print(f"Linked comments: {len(comments):,}")
    print(f"Planted high-volume posts: {len(planted_ids)}")
    print(f"Planted posts detected: {evaluation['planted_posts_detected']}")
    print(f"Planted posts missed: {evaluation['planted_posts_missed']}")
    print(f"Background posts flagged: {background_flags}")
    print("PASS: Every generated post count matches its linked comments.")


if __name__ == "__main__":
    main()
