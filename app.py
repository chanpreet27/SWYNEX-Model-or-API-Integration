from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "student_queries.csv"

@st.cache_resource
def load_model():
    df = pd.read_csv(DATA)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(df["text"].astype(str), df["label"].astype(str))
    return model

st.set_page_config(page_title="Student Query AI Classifier", page_icon="🎓")
st.title("🎓 Student Query AI Classifier")
st.write("Enter a student-support question. The model suggests a category.")
st.caption("Demo prototype · TF-IDF + Logistic Regression · Synthetic sample data")

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the dataset/model: {exc}")
    st.stop()

examples = ["How can I apply for a scholarship?", "I forgot my portal password",
            "What is the last date to pay semester fees?", "When will the exam timetable be published?",
            "Where is the college library?"]
choice = st.selectbox("Try an example", ["Choose an example…"] + examples)
question = st.text_area("Student question", value="" if choice == "Choose an example…" else choice,
                        placeholder="e.g. How can I apply for a scholarship?")
if st.button("Classify question", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        pred = model.predict([question.strip()])[0]
        probs = model.predict_proba([question.strip()])[0]
        st.subheader("Prediction")
        st.success(f"Suggested category: **{pred}**")
        st.metric("Indicative model score", f"{max(probs):.0%}")
        st.caption("This score is not a guarantee of correctness. Review uncertain predictions manually.")
        with st.expander("View category scores"):
            st.bar_chart(pd.DataFrame({"score": probs}, index=model.classes_))
with st.expander("Category guide"):
    st.markdown("- **Academics:** exams, classes, syllabus, assignments\n- **Fees:** tuition, payments, receipts, refunds\n- **Account:** login, password, profile, portal access\n- **Scholarship:** financial aid, eligibility, applications\n- **General:** campus facilities, offices, events")
st.divider()
st.caption("Uses 150 synthetic examples. Do not use predictions for official or high-impact decisions.")
