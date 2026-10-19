# SWYNEX — Model or API Integration
## Student Query Classification Prototype

This prototype integrates **scikit-learn**, a machine-learning library, to classify student-support questions into `Academics`, `Fees`, `Account`, `Scholarship`, or `General`. It uses TF-IDF text features and Logistic Regression, and includes a simple Streamlit interface.

### Run locally
1. Install Python 3.10+.
2. Open a terminal in the repository folder.
3. Create and activate a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Start the prototype: `streamlit run app.py`
6. Evaluate the model: `python src/evaluate_model.py`

### Example
Input: `How can I apply for a scholarship?`
Expected category: `Scholarship` (actual prediction is shown by the running app).

Input: `I forgot my portal password`
Expected category: `Account`.

### Data source, constraints, evaluation
The CSV contains 150 **synthetic demonstration examples** written for this project. They are not real student records. The dataset is small and may not represent real student language, typos, multilingual messages, or overlapping categories. The app's score is indicative, not a calibrated probability. Human review is required; do not use predictions for high-impact decisions.

`src/evaluate_model.py` uses a stratified 75/25 train/test split and reports accuracy, macro F1, precision, recall, and F1 by class. Results are illustrative only because the dataset is small and synthetic. A real deployment needs a larger, representative, human-reviewed dataset and an independent test set.
