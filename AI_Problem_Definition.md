# AI Problem Design — Student Query Classification

**Problem:** Student support questions need to be routed to the right service area. This prototype predicts a broad category; it does not answer questions or make administrative decisions.

**Intended user:** Student-support staff who need a first-pass category suggestion.

**Input/output:** Plain-text student question → one category (`Academics`, `Fees`, `Account`, `Scholarship`, `General`) and indicative category scores.

**Data source:** 150 synthetic examples created for this demonstration. No real student records or personal information are used.

**Model/library:** scikit-learn Pipeline with TF-IDF text features and Logistic Regression.

**Constraints:** The dataset is small and synthetic. Slang, typos, multilingual input, unseen topics, and ambiguous queries may reduce performance. Scores are not calibrated probabilities. Human review is required; no high-impact decisions should rely on this demo.

**Evaluation:** Reproducible stratified 75/25 train/test split, accuracy, macro F1, and per-class precision/recall/F1. Metrics are illustrative, not evidence of real-world performance.
