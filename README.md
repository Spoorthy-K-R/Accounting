# Financial Statement Analysis and Insights Platform

## Overview
This project automates the extraction, processing, and analysis of financial statements (10-Ks) from SEC EDGAR and Yahoo Finance for S&P 100 companies. It uses Python, NLP and large language models (LLMs) to generate financial insights, perform sentiment analysis on regulatory filings and identify key risks and opportunities for investors. (the NVIDIA folder is a sample analysis of NVIDIA financial data)

## Features
- Download and parse 10-K filings from the SEC EDGAR database
- Extract and structure financial tables (balance sheet, income statement, cash flow)
- Download historical financial data from Yahoo Finance
- Perform sentiment and keyword analysis on regulatory filings using the Loughran-McDonald financial dictionary
- Summarize multi-year financial trends and generate qualitative valuation using LLMs (uses Gemini, via LangChain)
- Output results as CSV, JSON and plain text summaries

## Project Structure
```
/Accounting
├── extract_financials.py                # Extracts financial tables from 10-K filings
├── sentiment_analysis_10k.py            # Performs sentiment/keyword analysis on 10-K text
├── NVIDIA/
│   ├── summarize_financials.py          # Summarizes a single year's financials of NVIDIA using LLM
│   ├── summarise-all-years.py           # Summarizes multi-year financials of NVIDIA using LLM
│   ├── output-csv/                      # Extracted financial tables (CSV) of NVIDIA
│   ├── output-txt/                      # LLM-generated analysis (TXT) on the financials of NVIDIA
│   └── 10K/                             # Downloaded 10-K filings of NVIDIA
├── Code 1. Download EDGAR Index.ipynb   # Download SEC index files
├── Code 2. Select 10-Ks.ipynb           # Select relevant 10-K filings
├── Code 3. Download 10-Ks.ipynb         # Download 10-K filings
├── Code 5. Obtain Financial Information from Yahoo Finance.ipynb # Download financials from Yahoo Finance
├── sp100_list.csv                       # list of S&P 100 companies
├── Loughran-McDonald_MasterDictionary_1993-2024.csv # Financial sentiment dictionary
```

## Installation
1. Clone the repository and navigate to the project directory.
2. Install the required Python packages:
   ```bash
   pip install pandas beautifulsoup4 yfinance nltk langchain langchain-google-genai
   ```
3. Download NLTK data (run once in Python):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
4. Obtain a Google Generative AI API key and set it as the `GOOGLE_API_KEY` environment variable for LLM-based scripts.

## Usage
- **Download SEC Index and 10-Ks:**
  - Use the provided Jupyter notebooks to download and select 10-K filings for S&P 100 companies.
- **Extract Financial Tables:**
  - Run `extract_financials.py` to parse 10-K filings and extract financial tables to CSV.
- **Sentiment Analysis:**
  - Run `sentiment_analysis_10k.py` to extract finance-related sentences and perform sentiment/keyword analysis. Outputs are saved as CSV and JSON.
- **Download Yahoo Finance Data:**
  - Use `Code 5. Obtain Financial Information from Yahoo Finance.ipynb` to fetch historical prices and financials.
- **Financial Analysis with LLMs:**
  - Run `NVIDIA/summarize_financials.py` for single-year analysis or `NVIDIA/summarise-all-years.py` for multi-year trend analysis. Results are saved in `output-txt/`. (the NVIDIA folder is a sample analysis of NVIDIA financial data)

## Example Output
Below is an excerpt from a 5-year LLM-generated analysis (see `NVIDIA/output-txt/NVIDIA_5yr_analysis.txt`):

## Acknowledgments
- SEC EDGAR for regulatory filings
- Yahoo Finance for financial data
- Loughran-McDonald for the financial sentiment dictionary
- LangChain and Google Generative AI for LLM-based analysis 