import csv
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "generated"
POST_COUNT = 2_000
COMMENTS_PER_POST = 3
SEED = 42

# Illustrative comments for testing the analysis pipeline.
EXAMPLES = [
    ("English", "What is the price of this {item}?"),
    ("Roman Urdu", "Is {item} ki price kya hai?"),
    ("Urdu", "اس سوٹ کی قیمت کیا ہے؟"),
    ("English", "Is the fabric of this {item} soft for summer?"),
    ("Roman Urdu", "Is {item} ka kapra garmi ke liye acha hai?"),
    ("Urdu", "کیا اس سوٹ کا کپڑا گرمیوں کے لیے اچھا ہے؟"),
    ("English", "Do you have this {item} in size {size}?"),
    ("Roman Urdu", "Ye {item} {size} size mein mil sakta hai?"),
    ("Urdu", "اس سوٹ کے کون کون سے سائز دستیاب ہیں؟"),
    ("English", "How many days will delivery to {city} take?"),
    ("Roman Urdu", "{city} mein delivery kitne din mein hogi?"),
    ("Urdu", "لاہور میں ڈیلیوری کتنے دن میں ہوگی؟"),
    ("English", "I ordered this {item} last week and am still waiting."),
    ("Roman Urdu", "Pichlay haftay order kiya tha abhi tak parcel nahi aya."),
    ("Urdu", "پچھلے ہفتے آرڈر کیا تھا ابھی تک پارسل نہیں آیا۔"),
    ("English", "Is there any discount on this {item}?"),
    ("Roman Urdu", "Is {item} par sale ya discount hai?"),
    ("Urdu", "کیا اس سوٹ پر کوئی رعایت ہے؟"),
    ("English", "Is this {item} available in your {city} store?"),
    ("Roman Urdu", "Ye {item} {city} ki branch mein available hai?"),
    ("Urdu", "کیا یہ سوٹ کراچی کی برانچ میں دستیاب ہے؟"),
    ("English", "I want to buy this {item}. How do I place an order?"),
    ("Roman Urdu", "Ye {item} lena hai order kaisay karun?"),
    ("Urdu", "میں یہ سوٹ خریدنا چاہتی ہوں آرڈر کیسے کروں؟"),
    ("English", "The colours of this {item} look beautiful."),
    ("Roman Urdu", "Is {item} ke colours bohat pyaray hain."),
    ("Urdu", "اس سوٹ کے رنگ بہت خوبصورت ہیں۔"),
    ("English", "The price is too high for this {item}."),
    ("Roman Urdu", "Is {item} ki qeemat bohat zyada hai."),
    ("Urdu", "اس سوٹ کی قیمت بہت زیادہ ہے۔"),
]

CITIES = ["Lahore", "Karachi", "Islamabad", "Multan", "Faisalabad"]
CONTENT_TYPES = [
    "Reel", "Product photo", "Influencer post",
    "Styling video", "Sale announcement", "Eid campaign",
]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng = random.Random(SEED)
    posts = []
    comments = []

    for number in range(1, POST_COUNT + 1):
        post_id = f"P{number:05d}"
        content_type = rng.choice(CONTENT_TYPES)
        video_content = content_type in {"Reel", "Styling video"}
        views = rng.randint(500, 150_000) if video_content else None
        likes = rng.randint(10, 5_000)
        if views is not None:
            likes = min(likes, views)

        posts.append({
            "post_id": post_id,
            "brand": rng.choice(["Demo Brand A", "Demo Brand B", "Demo Brand C"]),
            "platform": rng.choice(["Instagram", "TikTok", "Facebook"]),
            "campaign": rng.choice(["11.11 Collection", "12.12 Collection", "Eid Collection"]),
            "content_type": content_type,
            "product_category": "Women's lawn suits",
            "likes": likes,
            "comments_count": COMMENTS_PER_POST,
            "shares": rng.randint(0, 300) if rng.random() < 0.7 else None,
            "views": views,
            "days_since_post": rng.randint(1, 60),
            "data_origin": "synthetic",
            "source_url": "",
        })

        for _ in range(COMMENTS_PER_POST):
            language, template = rng.choice(EXAMPLES)
            text = template.format(
                item=rng.choice(["lawn suit", "printed suit", "embroidered suit"]),
                size=rng.choice(["small", "medium", "large"]),
                city=rng.choice(CITIES),
            )
            comments.append({
                "comment_id": f"C{len(comments) + 1:06d}",
                "post_id": post_id,
                "comment_text": text,
                "language_type": language,
                "data_origin": "synthetic",
            })

    post_ids = {post["post_id"] for post in posts}
    comment_ids = {comment["comment_id"] for comment in comments}
    counts = Counter(comment["post_id"] for comment in comments)

    if len(post_ids) != POST_COUNT or len(comment_ids) != len(comments):
        raise ValueError("Duplicate identifiers found.")
    if any(comment["post_id"] not in post_ids for comment in comments):
        raise ValueError("A comment refers to a missing post.")
    if any(counts[post["post_id"]] != post["comments_count"] for post in posts):
        raise ValueError("Post comment counts do not match the comment table.")
    if any(not comment["comment_text"].strip() for comment in comments):
        raise ValueError("Blank comments found.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "posts.csv", posts)
    write_csv(OUTPUT / "comments.csv", comments)

    manifest = {
        "data_origin": "synthetic",
        "seed": SEED,
        "posts": len(posts),
        "comments": len(comments),
        "unique_comment_texts": len({row["comment_text"] for row in comments}),
        "languages": dict(Counter(row["language_type"] for row in comments)),
        "template_count": len(EXAMPLES),
        "notes": [
            "Brands and engagement are illustrative.",
            "Comments are generated from templates and contain repeated phrases.",
            "Language labels come from the generator.",
            "Unavailable shares and views are blank, not zero.",
            "Each post has three comments in this starter dataset.",
            "Cluster findings describe this simulation, not actual customer behaviour.",
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("DATA GENERATION COMPLETE")
    print(f"Posts: {len(posts):,}")
    print(f"Comments: {len(comments):,}")
    print(f"Unique comment texts: {manifest['unique_comment_texts']:,}")
    print("PASS: IDs, post links, comment counts, and nonblank text verified.")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
