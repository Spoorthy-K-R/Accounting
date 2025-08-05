import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os

# --- User: Set these variables ---
api_key = os.getenv('GOOGLE_API_KEY')  # Or set your API key directly as a string

# List of CSVs for each year
years = ['22', '23', '24', '25']
base_path = '/Users/spoorthy/Projects/Accounting/NVIDIA/output-csv/'

balance_sheets = [f"{base_path}NVIDIA_{y}_balance_sheet.csv" for y in years]
cash_flows = [f"{base_path}NVIDIA_{y}_cash_flow.csv" for y in years]
income_statements = [f"{base_path}NVIDIA_{y}_income_statement.csv" for y in years]

# --- Combine all years for each statement ---
def concat_csvs(file_list):
    dfs = []
    for f in file_list:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Could not read {f}: {e}")
    return pd.concat(dfs, ignore_index=True)

df_balance = concat_csvs(balance_sheets)
df_cash = concat_csvs(cash_flows)
df_income = concat_csvs(income_statements)

balance_text = df_balance.to_string(index=False)
cash_text = df_cash.to_string(index=False)
income_text = df_income.to_string(index=False)

prompt_template = (
    """
    You are a financial analyst. Given the following financial statement tables for a company, perform a comprehensive analysis as follows:

    Financial Trends Analysis:
    - Summarize overall trends in revenue, profit, margins, debt, cash flow, and capital structure. Think of common size analysis and key financial ratios from a Financial Statement Analysis (FSA) course.
    - Identify any significant changes, anomalies, or turning points.
    - Spot potential red flags or opportunities (e.g., declining margins, rising leverage, growth acceleration).

    Insights and Explanations:
    - Explain the key drivers behind major changes
    - Interpret ratios or other computed metrics and provide context.
    - Assess risks and opportunities for investors.

    Valuation Advice:
    - Give a qualitative valuation perspective: Is the company undervalued, overvalued, or fairly valued based on its trends and financial performance? Should an investor buy, sell, or hold the company’s stock?
    - Provide supporting rationale (growth, profitability, risks, industry context).
    - Optionally, attempt a rough DCF, multiples-based, or scenario valuation and discuss any limitations.

    Here are the financial statements:

    Balance Sheet:
    {balance_text}

    Cash Flow Statement:
    {cash_text}

    Income Statement:
    {income_text}

    Please structure your response in three sections:
    1. Financial Trends Analysis
    2. Insights and Explanations
    3. Valuation Advice
    """
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyD8Lnshp4THbp7jYRF9IHFyxtyGAdEBzfA", temperature=0)
prompt = ChatPromptTemplate.from_template(prompt_template)
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.invoke({
    "balance_text": balance_text,
    "cash_text": cash_text,
    "income_text": income_text
})

print("\n--- AI Summary ---\n")
print(result['text'] if 'text' in result else result)

# Save output
out_file = f"/Users/spoorthy/Projects/Accounting/NVIDIA/output-txt/NVIDIA_analysis-new.txt"
with open(out_file, "w", encoding="utf-8") as out:
    out.write(result['text'] if 'text' in result else result)