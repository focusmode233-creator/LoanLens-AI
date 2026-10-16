import pandas as pd
import matplotlib.pyplot as plt

# Load our dataset
df = pd.read_csv("data/loan_dataset.csv")

# Clean column names by removing extra spaces
# and converting them to lowercase
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Display the actual column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Check whether the target column exists
if "loan_status" not in df.columns:
    print("\nERROR: The loan_status column was not found.")
    print("Please check the column names printed above.")

else:
    # Count approved and rejected applications
    print("\n--- LOAN STATUS COUNTS ---")
    print(df["loan_status"].value_counts())

    # Calculate percentages
    print("\n--- LOAN STATUS PERCENTAGES ---")
    print(df["loan_status"].value_counts(normalize=True) * 100)

    # Create a chart
    df["loan_status"].value_counts().plot(
        kind="bar",
        title="Distribution of Loan Status"
    )

    plt.xlabel("Loan Status")
    plt.ylabel("Number of Applications")
    plt.tight_layout()
    plt.show()