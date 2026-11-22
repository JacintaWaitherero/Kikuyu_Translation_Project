
from pathlib import Path
import pandas as pd

# Locate the raw data folder
raw_folder = Path("data/raw")

# Find Excel workbooks
excel_files = list(raw_folder.glob("*.xlsx"))

if not excel_files:
    print("No Excel workbook found in data/raw.")
else:
    for file in excel_files:
        print("\nWORKBOOK:", file.name)

        workbook = pd.ExcelFile(file)

        print("\nNumber of sheets:", len(workbook.sheet_names))

        for sheet in workbook.sheet_names:
            df = pd.read_excel(file, sheet_name=sheet)

            print("\n------------------------------")
            print("SHEET:", sheet)
            print("Number of rows:", len(df))
            print("Number of columns:", len(df.columns))
            print("Column names:", list(df.columns))

            print("\nFirst 3 rows:")
            print(df.head(3).to_string(index=False))
