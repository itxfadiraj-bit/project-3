# Pakistan Social Audience and Campaign Intelligence Engine

A Big Data Analytics project for discovering audience needs and campaign
opportunities in the Pakistani women's lawn-suit category.

## Business questions

- Which types of content attract attention?
- Which customer needs appear in comments?
- Which posts have unusual engagement patterns?
- Which comment groups should the brand address?
- Which marketing actions could be tested next?

## Planned methods

- Prepare linked social-post and comment tables.
- Preserve English, Roman Urdu, and Urdu comments.
- Use TF-IDF and clustering to discover similar comment groups.
- Compare content performance using available engagement metrics.
- Flag unusual posts for manual investigation.
- Show supporting examples for marketing recommendations.

## Data provenance

The initial demonstration will use clearly labeled synthetic records.
Generated engagement and comments are illustrative.
Any real imported data will require its source and collection information.

## Deliverables

A Streamlit dashboard, reproducible data preparation and analysis code,
downloadable findings, and a documented demo.

## Progress

- Project scope and folder structure created.
- Data preparation is the next milestone.

## Windows Quick Start

Install Python and Git, then open PowerShell and run:

```powershell
git clone https://github.com/engrmansoor304/Pakistan-Audience-IQ.git
cd .\Pakistan-Audience-IQ
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe .\run_pipeline.py
.\.venv\Scripts\python.exe -m streamlit run .\dashboard.py --server.port 8504
```

Open http://localhost:8504 in your browser.

## Dashboard walkthrough

- Audience Explorer filters comments by platform and language and shows cluster examples.
- Content Performance compares posts by platform, content type, and post age.
- Unusual Patterns shows engagement flags and controlled comment-volume validation.
- Marketing Tests presents provisional ideas with supporting comment IDs.

## Demonstration results

- Original dataset: 2,000 synthetic posts and 6,000 template-generated comments.
- Refined clustering: nine groups, each containing all three demonstration languages.
- Controlled volume test: 2,000 posts and 12,344 linked comments.
- Volume detection: 12 of 12 planted cases detected, zero missed, and 19 background posts flagged.

## Interpretation and limitations

The multilingual vocabulary is designed for this demonstration category.
Cluster names are provisional interpretations reviewed against cluster terms.
Silhouette measures text separation rather than marketing accuracy.

The volume test places 200 comments on planted posts and 2–8 on background
posts. Its results describe this controlled simulation. Engagement flags
require investigation, and marketing ideas require testing with actual data.

Generated datasets are excluded from Git. Run run_pipeline.py to recreate them.
The reports folder contains analysis and validation evidence.
