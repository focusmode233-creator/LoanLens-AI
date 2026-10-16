import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("data/loan_dataset.csv")

df.columns = (
    df.columns.str.strip().str.lower().str.replace(" ", "_")
)

print("Dataset loaded:", df.shape)

# 2. Separate features and target
X = df.drop(columns=["loan_status", "loan_id"])
y = df["loan_status"].str.strip().str.title()

# 3. Identify column types
categorical_columns = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 4. Prepare preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_columns),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
    ]
)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ),
}

results = []
trained_models = {}

# 7. Train and evaluate models
for name, classifier in models.items():

    print("\n" + "=" * 40)
    print("MODEL:", name)
    print("=" * 40)

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
    })

    trained_models[name] = model

# 8. Compare models
results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

# 9. Select the model with higher test accuracy
best_result = max(results, key=lambda item: item["Accuracy (%)"])
best_model_name = best_result["Model"]
best_model = trained_models[best_model_name]

# 10. Save selected model and results
joblib.dump(best_model, "loan_model.pkl")
results_df.to_csv("model_comparison.csv", index=False)

print("\nSelected model:", best_model_name)
print("Saved model: loan_model.pkl")
print("Saved comparison: model_comparison.csv")
print("\nPROJECT MODEL TRAINING COMPLETED!")