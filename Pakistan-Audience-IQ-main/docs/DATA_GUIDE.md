# Dataset Guide

The demonstration uses synthetic data for women's lawn suits.

## Original dataset

- posts.csv contains 2,000 posts with platform, campaign, content type,
  engagement metrics, and post age.
- comments.csv contains 6,000 linked comments.
- post_id links each comment to its post.
- English, Roman Urdu, and Urdu examples are generated from templates.
- Missing shares and views remain blank rather than becoming zero.

## Controlled volume dataset

data/generated/volume_demo contains a separate dataset with 2,000 posts.
Background posts receive 2–8 comments. Twelve planted posts receive 200.
The generator creates actual linked comments matching these counts.

The planted post IDs are stored separately for evaluation.
Run run_pipeline.py to recreate the generated data.
