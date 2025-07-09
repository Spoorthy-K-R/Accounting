import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os

# --- User: Set these variables ---
api_key = os.getenv('GOOGLE_API_KEY')  # Or set your API key directly as a string
balance_sheet_csv = '/Users/spoorthy/Projects/Accounting/NVIDIA/output-csv/NVIDIA_24_balance_sheet.csv'
cash_flow_csv = '/Users/spoorthy/Projects/Accounting/NVIDIA/output-csv/NVIDIA_24_cash_flow.csv'
income_statement_csv = '/Users/spoorthy/Projects/Accounting/NVIDIA/output-csv/NVIDIA_24_income_statement.csv'

# --- Load the CSVs ---
df_balance = pd.read_csv(balance_sheet_csv)
df_cash = pd.read_csv(cash_flow_csv)
df_income = pd.read_csv(income_statement_csv)

# --- Convert to string for prompt ---
balance_text = df_balance.to_string(index=False)
cash_text = df_cash.to_string(index=False)
income_text = df_income.to_string(index=False)

prompt_template = (
    """
    You are a financial analyst. Given the following financial statement tables for a company, summarize the overall trends in revenue, profit, margins, debt, cash flow, and capital structure. 
    
    Balance Sheet:
    {balance_text}

    Cash Flow Statement:
    {cash_text}

    Income Statement:
    {income_text}

    1. Please summarize the main trends and highlight any significant changes
    2. Based on these financials, what stands out as a potential risk or opportunity for investors?
    3. What factors could explain the fluctuation in operating margin?
    4. Can you provide a qualitative valuation assessment for this company based on its recent performance?

    Please reason step by step. Focus on key changes, growth or decline, and any notable patterns. If a metric is not present, mention that in your summary.

    """
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
prompt = ChatPromptTemplate.from_template(prompt_template)
chain = LLMChain(llm=llm, prompt=prompt)

# --- Call the LLM ---
result = chain.invoke({
    "balance_text": balance_text,
    "cash_text": cash_text,
    "income_text": income_text
})

print("\n--- AI Summary ---\n")
print(result['text'] if 'text' in result else result) 

file_name = balance_sheet_csv.split('/')
year = file_name[-1].split('_')[1]

out_file = f"/Users/spoorthy/Projects/Accounting/NVIDIA/output-txt/NVIDIA_{year}_analysis.txt"
with open(out_file, "w", encoding="utf-8") as out:
    out.write(result['text'] if 'text' in result else result)