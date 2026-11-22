
from pathlib import Path
import pandas as pd

# Folder containing the eight CSV files
processed_folder = Path("data/processed")

total_rows = 0
total_missing_english = 0
total_missing_kikuyu = 0
total_duplicates = 0

print("AGRICULTURAL DATA CLEANING REPORT")
print("=" * 45)

for file in sorted(processed_folder.glob("*.csv")):

    df = pd.read_csv(file, dtype=str, keep_default_na=False)

    # Remove extra spaces for checking only
    english = df["English"].str.strip()
    kikuyu = df["Kikuyu"].str.strip()

    # Check missing or empty sentences
    missing_english = english.eq("").sum()
    missing_kikuyu = kikuyu.eq("").sum()

    # Check duplicate English-Kikuyu sentence pairs
    checked_pairs = pd.DataFrame({
        "English": english.str.lower(),
        "Kikuyu": kikuyu.str.lower()
    })

    duplicates = checked_pairs.duplicated().sum()

    # Display results for this file
    print(f"\nFILE: {file.name}")
    print("Rows:", len(df))
    print("Missing English:", missing_english)
    print("Missing Kikuyu:", missing_kikuyu)
    print("Duplicate pairs:", duplicates)

    total_rows += len(df)
    total_missing_english += missing_english
    total_missing_kikuyu += missing_kikuyu
    total_duplicates += duplicates

print("\n" + "=" * 45)
print("OVERALL RESULTS")
print("Total rows:", total_rows)
print("Missing English:", total_missing_english)
print("Missing Kikuyu:", total_missing_kikuyu)
print("Duplicate pairs:", total_duplicates)
print("\nInspection completed. No files were changed.")
