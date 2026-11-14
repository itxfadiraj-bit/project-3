# Analysis Methods

## Preparation

Original comments are preserved. A cleaned text column normalizes whitespace,
case, and Unicode. Repeated phrases are retained and flagged.
Identifiers, post links, and comment counts are validated.

## Clustering

The baseline uses TF-IDF and KMeans on unique cleaned texts.
The refined version applies an inspectable multilingual vocabulary first.
Candidate cluster counts are compared using cosine silhouette scores.
Every original comment receives a cluster assignment.

Silhouette measures text separation. Cluster terms, examples, and language
composition require review before assigning marketing interpretations.
The baseline and refined versions use different feature spaces.

## Unusual engagement

Posts are compared within platform, content type, and age band.
A robust score uses log-transformed engagement, peer medians, and
median absolute deviation. Groups require at least 15 posts.
Absolute scores above 3.5 trigger review.

Controlled volume validation records planted detections, missed cases,
and background flags. Results apply to the synthetic test.
