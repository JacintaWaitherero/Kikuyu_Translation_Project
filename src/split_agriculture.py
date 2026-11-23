
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the cleaned agricultural dataset
input_file = Path("data/processed/cleaned/agriculture_cleaned.csv")

df = pd.read_csv(input_file)

# Check required columns
required_columns = ["English", "Kikuyu", "Category"]

if not all(col in df.columns for col in required_columns):
    raise ValueError("Required columns are missing.")

# First separate 20% for validation and testing
train, temporary = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Category"]
)

# Divide the remaining 20% equally
validation, test = train_test_split(
    temporary,
    test_size=0.50,
    random_state=42,
    stratify=temporary["Category"]
)

# Create output folder
output_folder = Path("data/processed/splits")
output_folder.mkdir(parents=True, exist_ok=True)

# Save the three datasets
train.to_csv(
    output_folder / "train.csv",
    index=False,
    encoding="utf-8-sig"
)

validation.to_csv(
    output_folder / "validation.csv",
    index=False,
    encoding="utf-8-sig"
)

test.to_csv(
    output_folder / "test.csv",
    index=False,
    encoding="utf-8-sig"
)

# Display results
print("\nDATA SPLITTING COMPLETED")
print("=" * 40)
print("Original pairs:", len(df))
print("Training pairs:", len(train))
print("Validation pairs:", len(validation))
print("Testing pairs:", len(test))
print("Total after splitting:", len(train) + len(validation) + len(test))

print("\nCategory distribution:")
print(pd.DataFrame({
    "Train": train["Category"].value_counts(),
    "Validation": validation["Category"].value_counts(),
    "Test": test["Category"].value_counts()
}).fillna(0).astype(int))

print("\nFiles saved in:", output_folder)
