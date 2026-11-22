
from pathlib import Path
import pandas as pd

processed_folder = Path("data/processed")

print("MISSING TRANSLATIONS")
print("=" * 60)

for file in sorted(processed_folder.glob("*.csv")):
    df = pd.read_csv(file, dtype=str, keep_default_na=False)

    english = df["English"].str.strip()
    kikuyu = df["Kikuyu"].str.strip()

    missing = (english.eq("")) | (kikuyu.eq(""))

    if missing.any():
        print(f"\nFILE: {file.name}")

        for index, row in df[missing].iterrows():
            print(f"\nCSV row number: {index + 2}")
            print("English:", repr(row["English"]))
            print("Kikuyu:", repr(row["Kikuyu"]))


print("\n\nDUPLICATE SENTENCE PAIRS")
print("=" * 60)

for file in sorted(processed_folder.glob("*.csv")):
    df = pd.read_csv(file, dtype=str, keep_default_na=False)

    english = df["English"].str.strip()
    kikuyu = df["Kikuyu"].str.strip()

    # Only check rows where both translations exist
    complete = english.ne("") & kikuyu.ne("")

    pairs = pd.DataFrame({
        "English": english.str.lower(),
        "Kikuyu": kikuyu.str.lower()
    })

    duplicate_mask = pairs[complete].duplicated(keep=False)

    if duplicate_mask.any():
        print(f"\nFILE: {file.name}")

        duplicate_indices = pairs[complete].index[duplicate_mask]

        for index in duplicate_indices:
            print(f"\nCSV row number: {index + 2}")
            print("English:", df.loc[index, "English"])
            print("Kikuyu:", df.loc[index, "Kikuyu"])

print("\nInspection completed. No files were changed.")
