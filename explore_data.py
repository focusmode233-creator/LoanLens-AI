import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("data/loan_dataset.csv")

# Step 2: Display the first five rows
print("\n--- FIRST FIVE ROWS ---")
print(df.head())

# Step 3: Display the dataset dimensions
print("\n--- DATASET SIZE ---")
print("Rows and columns:", df.shape)

# Step 4: Display all column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Step 5: Display information about each column
print("\n--- DATASET INFORMATION ---")
df.info()

# Step 6: Count missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Step 7: Display basic numerical statistics
print("\n--- NUMERICAL SUMMARY ---")
print(df.describe())