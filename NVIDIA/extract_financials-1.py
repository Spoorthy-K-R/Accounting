from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

files = ["/Users/spoorthy/Projects/Accounting/NVIDIA/10K/0001045810-20-000010.txt",
         "/Users/spoorthy/Projects/Accounting/NVIDIA/10K/0001045810-21-000010.txt",
         "/Users/spoorthy/Projects/Accounting/NVIDIA/10K/0001045810-22-000036.txt",
         "/Users/spoorthy/Projects/Accounting/NVIDIA/10K/0001045810-23-000017.txt",
         "/Users/spoorthy/Projects/Accounting/NVIDIA/10K/0001045810-24-000029.txt"]

section_headers = {
    'income_statement': [
        'consolidated statements of operations',
        'consolidated statements of income',
        'consolidated statement of operations',
        'consolidated statement of income'
    ],
    'balance_sheet': [
        'consolidated balance sheets',
        'consolidated balance sheet'
    ],
    'cash_flow': [
        'consolidated statements of cash flows',
        'consolidated statement of cash flows'
    ]
}

for input_file in files:
    year = input_file.split('-')[1]
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, 'html.parser')

    tables = soup.find_all('table')
    print(f"{len(tables)} tables found in {input_file}")
    print(tables[0])

    for section, headers in section_headers.items():
        found = False
        for tag in soup.find_all(text=True):
            if any(header in tag.lower() for header in headers):
                # Find the next table after the header
                next_table = tag.find_parent().find_next('table')
                if next_table:
                    df = pd.read_html(StringIO(str(next_table)))[0]
                    csv_file = f"/Users/spoorthy/Projects/Accounting/NVIDIA/output-csv/NVIDIA_{year}_{section}.csv"
                    df.to_csv(csv_file, index=False)
                    print(f"Saved {section} to {csv_file}")
                    found = True
                    break
        if not found:
            print(f"{section} not found in {input_file}")