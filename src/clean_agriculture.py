
from pathlib import Path
import pandas as pd

# Input and output folders
input_folder = Path("data/processed")
output_folder = Path("data/processed/cleaned")
output_folder.mkdir(parents=True, exist_ok=True)

cleaned_datasets = []
incomplete_rows = []
duplicate_rows = []
summary = []

# Read the eight original CSV files
for file in sorted(input_folder.glob("*.csv")):

    df = pd.read_csv(file, dtype=str, keep_default_na=False)

    # Clean unnecessary spaces without changing the wording
    df["English"] = df["English"].apply(lambda x: " ".join(x.split()))
    df["Kikuyu"] = df["Kikuyu"].apply(lambda x: " ".join(x.split()))

    # Identify the agricultural category
    if "Category" not in df.columns:
        df["Category"] = file.stem

    original_count = len(df)

    # Identify incomplete sentence pairs
    incomplete = (
        df["English"].eq("") |
        df["Kikuyu"].eq("")
    )

    removed_incomplete = df[incomplete].copy()
    removed_incomplete["Source_File"] = file.name
    removed_incomplete["Original_Row"] = removed_incomplete.index + 2

    incomplete_rows.append(removed_incomplete)

    # Keep only complete sentence pairs
    complete = df[~incomplete].copy()

    # Create temporary keys for duplicate checking
    complete["_English_key"] = complete["English"].str.casefold()
    complete["_Kikuyu_key"] = complete["Kikuyu"].str.casefold()

    # Identify repeated pairs, keeping the first occurrence
    duplicate_mask = complete.duplicated(
        subset=["_English_key", "_Kikuyu_key"],
        keep="first"
    )

    removed_duplicates = complete[duplicate_mask].copy()
    removed_duplicates["Source_File"] = file.name
    removed_duplicates["Original_Row"] = removed_duplicates.index + 2

    duplicate_rows.append(removed_duplicates)

    # Remove the repeated copies
    cleaned = complete[~duplicate_mask].copy()

    # Remove temporary checking columns
    cleaned = cleaned.drop(
        columns=["_English_key", "_Kikuyu_key"]
    )

    cleaned_datasets.append(cleaned)

    summary.append({
        "Category": file.stem,
        "Original_Rows": original_count,
        "Incomplete_Rows_Removed": int(incomplete.sum()),
        "Duplicate_Rows_Removed": int(duplicate_mask.sum()),
        "Cleaned_Rows": len(cleaned)
    })

# Combine all eight cleaned datasets
combined = pd.concat(cleaned_datasets, ignore_index=True)

# Save the combined agricultural dataset
combined.to_csv(
    output_folder / "agriculture_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)

# Save records of removed rows
if incomplete_rows:
    pd.concat(incomplete_rows, ignore_index=True).to_csv(
        output_folder / "incomplete_rows_removed.csv",
        index=False,
        encoding="utf-8-sig"
    )

if duplicate_rows:
    pd.concat(duplicate_rows, ignore_index=True).to_csv(
        output_folder / "duplicate_rows_removed.csv",
        index=False,
        encoding="utf-8-sig"
    )

# Save the cleaning summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    output_folder / "cleaning_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# Identify identical pairs appearing in different categories.
# We report these but do not remove them yet.
combined["_English_key"] = combined["English"].str.casefold()
combined["_Kikuyu_key"] = combined["Kikuyu"].str.casefold()

cross_duplicates = combined[
    combined.duplicated(
        subset=["_English_key", "_Kikuyu_key"],
        keep=False
    )
].copy()

cross_category = (
    cross_duplicates.groupby(
        ["_English_key", "_Kikuyu_key"]
    )["Category"].nunique()
)

cross_category = cross_category[cross_category > 1].index

if len(cross_category) > 0:
    cross_report = cross_duplicates.set_index(
        ["_English_key", "_Kikuyu_key"]
    ).loc[cross_category].reset_index()
else:
    cross_report = pd.DataFrame(
        columns=["English", "Kikuyu", "Category"]
    )

cross_report = cross_report.drop(
    columns=["_English_key", "_Kikuyu_key"],
    errors="ignore"
)

cross_report.to_csv(
    output_folder / "cross_category_duplicates_to_review.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nCLEANING COMPLETED")
print("=" * 45)
print(summary_df.to_string(index=False))

print("\nTOTAL ORIGINAL ROWS:", sum(s["Original_Rows"] for s in summary))
print("INCOMPLETE ROWS REMOVED:", sum(s["Incomplete_Rows_Removed"] for s in summary))
print("DUPLICATES REMOVED:", sum(s["Duplicate_Rows_Removed"] for s in summary))
print("CLEANED ROWS:", len(combined))
print("CROSS-CATEGORY DUPLICATES TO REVIEW:", len(cross_report))

print("\nFiles saved in:", output_folder)
