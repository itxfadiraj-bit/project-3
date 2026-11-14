import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "generated"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Pakistan Audience IQ", page_icon="📊", layout="wide")

# Provisional interpretations reviewed against this run's reported cluster terms.
REVIEW = {
    0: ("Availability inquiries", "available", "Test a branch and stock availability post."),
    1: ("Price discussion", "price", "Review price inquiries versus complaints before testing price-focused content."),
    2: ("Sizing inquiries", "size", "Test a clear size guide with product measurements."),
    3: ("Order delay concerns", "week", "Review delayed-order examples and test clearer order-status communication."),
    4: ("Fabric questions", "fabric", "Test a fabric close-up and summer-comfort explanation."),
    5: ("Delivery timing", "delivery", "Test city-specific delivery time information."),
    6: ("Purchase inquiries", "buy", "Test a clearer ordering guide and call to action."),
    7: ("Discount inquiries", "discount", "Test a promotion explanation after checking campaign margins."),
    8: ("Colour appreciation", "colour", "Test a colour-focused product presentation."),
}

required_files = [
    DATA / "refined_clustered_comments.csv",
    DATA / "scored_posts.csv",
    REPORTS / "refined_cluster_summary.csv",
    REPORTS / "content_performance.csv",
    REPORTS / "unusual_posts.csv",
    REPORTS / "unusual_comment_volume.csv",
    REPORTS / "volume_validation.json",
]
missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]

st.title("Pakistan Audience IQ")
st.caption("Social audience and campaign intelligence · Women's lawn suits")
st.info("Demonstration using synthetic posts and template-generated comments.")

if missing:
    st.error("Complete the analysis scripts first. Missing files:")
    st.write(missing)
    st.stop()

comments = pd.read_csv(DATA / "refined_clustered_comments.csv", encoding="utf-8-sig")
posts = pd.read_csv(DATA / "scored_posts.csv", encoding="utf-8-sig")
clusters = pd.read_csv(REPORTS / "refined_cluster_summary.csv", encoding="utf-8-sig")
performance = pd.read_csv(REPORTS / "content_performance.csv")
unusual = pd.read_csv(REPORTS / "unusual_posts.csv", encoding="utf-8-sig")
volume_flags = pd.read_csv(REPORTS / "unusual_comment_volume.csv")
validation = json.loads((REPORTS / "volume_validation.json").read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Explore the audience")
    platform = st.selectbox("Platform", ["All", *sorted(posts["platform"].unique())])
    language = st.selectbox(
        "Comment language", ["All", *sorted(comments["language_type"].unique())]
    )
    st.divider()
    st.caption("Audience analysis uses the original 6,000-comment dataset.")
    st.caption("Volume validation uses a separate controlled dataset.")

selected_posts = posts if platform == "All" else posts.loc[posts["platform"].eq(platform)]
selected_comments = comments.loc[comments["post_id"].isin(selected_posts["post_id"])]
if language != "All":
    selected_comments = selected_comments.loc[selected_comments["language_type"].eq(language)]

a, b, c = st.columns(3)
a.metric("Posts in platform selection", f"{len(selected_posts):,}")
b.metric("Comments after filters", f"{len(selected_comments):,}")
c.metric("Audience clusters in selection", selected_comments["cluster_id"].nunique())

audience, content, anomalies, actions = st.tabs(
    ["Audience Explorer", "Content Performance", "Unusual Patterns", "Marketing Tests"]
)

with audience:
    st.subheader("Discover common audience needs")
    st.caption(
        "Dictionary-assisted TF-IDF and KMeans. Cluster names below are provisional "
        "human interpretations. Review the original examples."
    )

    if selected_comments.empty:
        st.warning("No comments match these filters.")
    else:
        sizes = selected_comments.groupby("cluster_id").size().rename("comments").reset_index()
        sizes["cluster"] = sizes["cluster_id"].map(lambda number: f"Cluster {number}")
        st.plotly_chart(
            px.bar(sizes, x="cluster", y="comments", color_discrete_sequence=["#2563EB"]),
            width="stretch",
        )
        choice = st.selectbox("Inspect a cluster", sorted(selected_comments["cluster_id"].unique()))
        details = clusters.loc[clusters["cluster_id"].eq(choice)].iloc[0]
        reviewed = REVIEW.get(int(choice))
        matches_review = reviewed is not None and reviewed[1] in str(details["top_terms"]).split(", ")

        if matches_review:
            st.markdown(f"**Provisional interpretation — {reviewed[0]}**")
        st.write("Common terms", details["top_terms"])

        examples = selected_comments.loc[selected_comments["cluster_id"].eq(choice)]
        st.dataframe(
            examples[["comment_id", "post_id", "comment_text", "language_type"]].head(100),
            hide_index=True,
        )
        st.download_button(
            "Download selected cluster comments",
            examples.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"cluster_{choice}_comments.csv", mime="text/csv",
        )

with content:
    st.subheader("Compare similar content")
    st.caption(
        "Groups use platform, content type, and post age. The sidebar language "
        "filter applies to comments. It does not filter post performance."
    )
    comparison = performance if platform == "All" else performance.loc[
        performance["platform"].eq(platform)
    ]
    st.dataframe(comparison, hide_index=True)
    st.caption(
        "Shares and views remain missing when unavailable. Likes per 1,000 views "
        "uses posts with positive view counts."
    )

with anomalies:
    st.subheader("Unusual likes within comparable groups")
    selected_flags = unusual if platform == "All" else unusual.loc[unusual["platform"].eq(platform)]
    st.metric("Posts flagged in platform selection", len(selected_flags))
    st.caption("These are statistical review flags. Investigate their context before taking action.")
    st.dataframe(
        selected_flags[
            ["post_id", "platform", "content_type", "age_band", "likes", "likes_robust_score"]
        ], hide_index=True,
    )

    st.divider()
    st.subheader("Controlled comment-volume validation")
    st.caption("Full test dataset, independent of the sidebar filters.")
    x, y, z = st.columns(3)
    x.metric("Planted posts detected", f"{validation['planted_posts_detected']} / {validation['planted_posts']}")
    y.metric("Planted posts missed", validation["planted_posts_missed"])
    z.metric("Background posts flagged", validation["background_posts_flagged"])
    st.write(
        f"Background posts eligible for review: {validation['eligible_background_posts']:,}. "
        "This test uses 200 comments on planted posts and 2–8 on background posts."
    )
    st.caption("The result measures this obvious synthetic test. It does not establish real-world accuracy.")
    st.dataframe(
        volume_flags[
            ["post_id", "platform", "content_type", "comments_count", "comment_volume_score"]
        ], hide_index=True,
    )

with actions:
    st.subheader("Evidence for marketing tests")
    st.caption(
        "Ideas depend on provisional cluster interpretations and the synthetic "
        "sample. Validate them with actual audience feedback before use."
    )
    ideas = []
    for cluster_id, members in selected_comments.groupby("cluster_id"):
        reviewed = REVIEW.get(int(cluster_id))
        details = clusters.loc[clusters["cluster_id"].eq(cluster_id)]
        if reviewed is None or details.empty:
            continue
        if reviewed[1] not in str(details.iloc[0]["top_terms"]).split(", "):
            continue
        ideas.append({
            "cluster_id": int(cluster_id),
            "provisional_need": reviewed[0],
            "supporting_comments": len(members),
            "example_comment_ids": ", ".join(members["comment_id"].head(3)),
            "suggested_test": reviewed[2],
        })

    if ideas:
        ideas = pd.DataFrame(ideas).sort_values("supporting_comments", ascending=False)
        st.dataframe(ideas, hide_index=True)
        st.download_button(
            "Download marketing test ideas",
            ideas.to_csv(index=False).encode("utf-8-sig"),
            file_name="marketing_test_ideas.csv", mime="text/csv",
        )
    else:
        st.info("No reviewed cluster interpretation matches this selection.")
