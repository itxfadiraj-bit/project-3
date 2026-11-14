from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
REPORTS = ROOT / "reports"
MIN_PEERS = 15
THRESHOLD = 3.5


def robust_score(values):
    median = values.median()
    mad = (values - median).abs().median()
    if mad == 0:
        return pd.Series(np.nan, index=values.index)
    return 0.6745 * (values - median) / mad


def main():
    posts = pd.read_csv(DATA / "posts.csv", encoding="utf-8-sig")
    required = {
        "post_id", "platform", "content_type", "likes",
        "comments_count", "shares", "views", "days_since_post",
    }
    if not required.issubset(posts.columns):
        raise SystemExit("The posts table is missing required columns.")

    if posts["post_id"].isna().any() or posts["post_id"].duplicated().any():
        raise SystemExit("Post identifiers must be nonblank and unique.")

    metrics = ["likes", "comments_count", "shares", "views", "days_since_post"]
    for column in metrics:
        posts[column] = pd.to_numeric(posts[column], errors="raise")
        if posts[column].dropna().lt(0).any():
            raise SystemExit(f"Negative values found in {column}.")

    if posts[["likes", "comments_count", "days_since_post"]].isna().any().any():
        raise SystemExit("Likes, comment counts, and post ages must be available.")

    posts["age_band"] = pd.cut(
        posts["days_since_post"],
        bins=[-1, 7, 30, np.inf],
        labels=["0-7 days", "8-30 days", "31+ days"],
    ).astype(str)

    # Shares stay missing when unavailable. Video rates use positive views only.
    posts["likes_per_1000_views"] = (
        1000 * posts["likes"] / posts["views"].where(posts["views"].gt(0))
    )

    peer_columns = ["platform", "content_type", "age_band"]
    groups = posts.groupby(peer_columns, observed=True)
    posts["peer_posts"] = groups["post_id"].transform("size")

    # Log scale reduces the influence of very large absolute counts.
    posts["log_likes"] = np.log1p(posts["likes"])
    posts["likes_robust_score"] = groups["log_likes"].transform(robust_score)

    posts["review_eligible"] = (
        posts["peer_posts"].ge(MIN_PEERS)
        & posts["likes_robust_score"].notna()
    )
    posts["unusual_likes"] = (
        posts["review_eligible"]
        & posts["likes_robust_score"].abs().gt(THRESHOLD)
    )

    summary = groups.agg(
        posts=("post_id", "size"),
        median_likes=("likes", "median"),
        median_comments=("comments_count", "median"),
        shares_available=("shares", "count"),
        median_shares=("shares", "median"),
        views_available=("views", "count"),
        median_likes_per_1000_views=("likes_per_1000_views", "median"),
    ).reset_index().round(2)

    review = posts.loc[posts["unusual_likes"]].copy()
    review["score_magnitude"] = review["likes_robust_score"].abs()
    review = review.sort_values("score_magnitude", ascending=False)

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS / "content_performance.csv", index=False)
    posts.drop(columns="log_likes").to_csv(
        DATA / "scored_posts.csv", index=False, encoding="utf-8-sig"
    )
    review.drop(columns="log_likes").to_csv(
        REPORTS / "unusual_posts.csv", index=False, encoding="utf-8-sig"
    )

    print("CAMPAIGN ANALYSIS COMPLETE")
    print(f"Posts analyzed: {len(posts):,}")
    print(f"Comparable groups: {len(summary):,}")
    print(f"Posts eligible for unusual-likes review: {posts['review_eligible'].sum():,}")
    print(f"Posts flagged for review: {len(review):,}")
    print(f"Shares available: {posts['shares'].notna().sum():,}")
    print(f"Positive views available: {posts['views'].gt(0).sum():,}")

    if posts["comments_count"].nunique() == 1:
        print("Starter data has constant comment counts; comment-volume anomalies cannot be evaluated.")
    print("Flags indicate unusual likes within comparable groups and require review.")
    print(f"Reports saved to: {REPORTS}")


if __name__ == "__main__":
    main()
