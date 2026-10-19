from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "student_queries.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"])
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Test examples: {len(y_test)}")
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"Macro F1: {f1_score(y_test, pred, average='macro'):.3f}")
print(classification_report(y_test, pred, zero_division=0))
print("Metrics are illustrative only: this is a small synthetic demo dataset.")
