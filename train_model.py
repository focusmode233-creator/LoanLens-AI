import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------------

df = pd.read_csv("data/loan_dataset.csv")

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# --------------------------------------------------

# Remove loan_id because it is an identifier,
# not a meaningful applicant feature.
X = df.drop(columns=["loan_status", "loan_id"])

# This is what the model will learn to predict.
y = df["loan_status"].str.strip().str.title()

print("\nFeatures:", X.columns.tolist())
print("Target classes:", y.unique())


# --------------------------------------------------
# 3. IDENTIFY COLUMN TYPES
# --------------------------------------------------

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)


# --------------------------------------------------
# 4. PREPROCESS THE DATA
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_columns,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
    ]
)


# --------------------------------------------------
# 5. SPLIT INTO TRAINING AND TESTING DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 6. CREATE AND TRAIN THE MODEL
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)

print("\nModel training completed!")


# --------------------------------------------------
# 7. EVALUATE THE MODEL
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print("Class order:", model.named_steps["classifier"].classes_)
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 8. SAVE THE TRAINED MODEL
# --------------------------------------------------

import joblib

joblib.dump(model, "loan_model.pkl")

print("\nModel saved as loan_model.pkl")