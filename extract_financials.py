import re
from bs4 import BeautifulSoup
import pandas as pd

# Path to your 10-K file
files = ["/Users/spoorthy/Projects/Accounting/NVIDIA/10K/0001045810-20-000010.txt",
         "/Users/spoorthy/Projects/Accounting/NVIDIA/10K/0001045810-21-000010.txt",
         "/Users/spoorthy/Projects/Accounting/NVIDIA/10K/0001045810-22-000036.txt",
         "/Users/spoorthy/Projects/Accounting/NVIDIA/10K/0001045810-23-000017.txt",
         "/Users/spoorthy/Projects/Accounting/NVIDIA/10K/0001045810-24-000029.txt"]

# Define possible section headers (expand as needed)
sections = {
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

def extract_section(lines, headers):
    # Find the start of the section
    start_idx = None
    for i, line in enumerate(lines):
        for header in headers:
            if header in line.lower():
                start_idx = i
                break
        if start_idx is not None:
            break
    if start_idx is None:
        return None  # Section not found

    # Extract lines after the header
    extracted = []
    for line in lines[start_idx+1:]:
        # Stop if we hit another section header or a long separator line
        if any(h in line.lower() for hlist in sections.values() for h in hlist):
            break
        if re.match(r'^[\-=\s]{10,}$', line):  # long line of dashes/spaces
            break
        extracted.append(line)
        if len(extracted) > 100:  # Avoid runaway extraction
            break
    return "".join(extracted).strip()

txt_files=[]

for input_file in files:
    file_name = input_file.split('/')
    year = file_name[-1].split('-')[1]
    print(year)
    # Read the file in manageable chunks
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for section, headers in sections.items():
        content = extract_section(lines, headers)
        if content:
            out_file = f"/Users/spoorthy/Projects/Accounting/NVIDIA/output-txt/NVIDIA_{year}_{section}.txt"
            with open(out_file, "w", encoding="utf-8") as out:
                out.write(content)
            print(f"Extracted {section} to {out_file}")
        else:
            print(f"{section} not found.") 
        txt_files.append(out_file)

print(txt_files)
for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Step 2: Find all tables in the HTML
    tables = soup.find_all('table')

    csv_file = f"/Users/spoorthy/Projects/Accounting/NVIDIA/output-csv/{txt_file.split('.')[0].split('/')[-1]}.csv"
    # Step 3: Loop through tables and save each as a CSV
    for i, table in enumerate(tables):
        df = pd.read_html(str(table))[0]  # Convert HTML table to DataFrame
        df.to_csv(csv_file, index=False)
        print(f"Saved table {i+1} to NVIDIA_20_balance_sheet_table_{i+1}.csv")
