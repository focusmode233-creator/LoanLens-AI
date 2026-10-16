"""Train, select, and evaluate LoanLens AI classifiers without test-set selection.

Run from the project root: python compare_models.py
The final test set is held out until model selection is complete.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_dataset.csv"
MODEL_PATH = ROOT / "loan_model.pkl"
COMPARISON_PATH = ROOT / "model_comparison.csv"
REPORT_PATH = ROOT / "classification_report.csv"
CONFUSION_PATH = ROOT / "confusion_matrix.csv"

# 1. Load and validate data
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
required = {"loan_status", "loan_id"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
if df["loan_status"].isna().any():
    raise ValueError("Target column loan_status contains missing values.")

X = df.drop(columns=["loan_status", "loan_id"])
y = df["loan_status"].astype(str).str.strip().str.title()
if y.nunique() < 2:
    raise ValueError("At least two target classes are required.")

categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical = X.select_dtypes(include=["number"]).columns.tolist()

# Preprocessing is fitted inside each pipeline/fold, avoiding preprocessing leakage.
def make_pipeline(classifier):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), numerical),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessing", preprocessor), ("classifier", classifier)])

# 2. Split once: the final test set is not used for model selection.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Training records: {len(X_train)} | Final test records: {len(X_test)}")
print("Target distribution:")
print(y.value_counts().to_string())

models = {
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=2000, random_state=42)),
    "Random Forest": make_pipeline(RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
    )),
}

# 3. Select using stratified CV on training data only.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name, pipeline in models.items():
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring={"accuracy": "accuracy", "precision_macro": "precision_macro",
                 "recall_macro": "recall_macro", "f1_macro": "f1_macro"},
        n_jobs=-1, return_train_score=False,
    )
    row = {
        "Model": name,
        "CV Accuracy Mean (%)": round(scores["test_accuracy"].mean() * 100, 2),
        "CV Accuracy Std (percentage points)": round(scores["test_accuracy"].std() * 100, 2),
        "CV Precision Macro (%)": round(scores["test_precision_macro"].mean() * 100, 2),
        "CV Recall Macro (%)": round(scores["test_recall_macro"].mean() * 100, 2),
        "CV F1 Macro (%)": round(scores["test_f1_macro"].mean() * 100, 2),
    }
    cv_rows.append(row)
    print(f"\n{name} — 5-fold training CV")
    for key, value in row.items():
        if key != "Model": print(f"  {key}: {value}")

comparison = pd.DataFrame(cv_rows).sort_values(
    ["CV Accuracy Mean (%)", "CV F1 Macro (%)"], ascending=False
).reset_index(drop=True)
best_name = comparison.loc[0, "Model"]
best_model = models[best_name]

# 4. Fit both candidates on training data and evaluate on the untouched test set.
# Test metrics are reported for transparency, but do NOT determine the winner.
labels = sorted(y.unique())
test_predictions = {}
test_rows = {}
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    test_predictions[name] = pred
    test_rows[name] = {
        "Accuracy (%)": round(accuracy_score(y_test, pred) * 100, 2),
        "Final Test Precision Macro (%)": round(precision_score(y_test, pred, average="macro", zero_division=0) * 100, 2),
        "Final Test Recall Macro (%)": round(recall_score(y_test, pred, average="macro", zero_division=0) * 100, 2),
        "Final Test F1 Macro (%)": round(f1_score(y_test, pred, average="macro", zero_division=0) * 100, 2),
    }

best_model = models[best_name]
y_pred = test_predictions[best_name]
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
for metric in ["Accuracy (%)", "Final Test Precision Macro (%)", "Final Test Recall Macro (%)", "Final Test F1 Macro (%)"]:
    comparison[metric] = comparison["Model"].map({name: values[metric] for name, values in test_rows.items()})
comparison.to_csv(COMPARISON_PATH, index=False)

report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose().reset_index(names="Class / Metric")
report_df.to_csv(REPORT_PATH, index=False)
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels])
cm_df.to_csv(CONFUSION_PATH, index=True, index_label="Actual / Predicted")

joblib.dump(best_model, MODEL_PATH)

print("\nMODEL SELECTION (training data only)")
print(comparison[["Model", "CV Accuracy Mean (%)", "CV Accuracy Std (percentage points)", "CV F1 Macro (%)"]].to_string(index=False))
print(f"\nSelected model: {best_name}")
print("\nFINAL HELD-OUT TEST RESULTS")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro precision: {precision_macro * 100:.2f}%")
print(f"Macro recall: {recall_macro * 100:.2f}%")
print(f"Macro F1-score: {f1_macro * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
print("Confusion matrix (class order: " + ", ".join(labels) + "):")
print(cm)
print(f"\nSaved: {MODEL_PATH.name}, {COMPARISON_PATH.name}, {REPORT_PATH.name}, {CONFUSION_PATH.name}")
