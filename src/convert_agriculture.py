
from pathlib import Path
import pandas as pd

# Locate the original Excel workbook
raw_folder = Path("data/raw")
processed_folder = Path("data/processed")

# Create the output folder if it does not exist
processed_folder.mkdir(parents=True, exist_ok=True)

# Find the Excel workbook
excel_files = list(raw_folder.glob("*.xlsx"))

if len(excel_files) != 1:
    print("Expected exactly one Excel workbook in data/raw.")
else:
    excel_file = excel_files[0]

    # Read the workbook
    workbook = pd.ExcelFile(excel_file)

    # Convert every sheet separately
    for sheet in workbook.sheet_names:

        df = pd.read_excel(excel_file, sheet_name=sheet)

        # Extract the agricultural category
        category = sheet.replace("Sentence Pairs - ", "")

        # Add a category column
        df["Category"] = category

        # Create a filename
        filename = category.lower().replace(" ", "_") + ".csv"

        # Save the sheet as CSV
        output_path = processed_folder / filename

        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"Saved: {filename} | Rows: {len(df)}")

    print("\nConversion completed!")
