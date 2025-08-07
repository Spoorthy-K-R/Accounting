import os
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import shutil
import numpy as np
import json 
import requests
import os
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import zipfile
from io import BytesIO
import time
import glob
import re
import shutil

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AIMessage

CACHE_DIR = 'llm_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

SECTION_REGEX_PATTERNS = {
    'income_statement': r'consolidated statements of (operations|income).*?<table.*?>(.*?)</table>',
    'balance_sheet': r'consolidated balance sheets.*?<table.*?>(.*?)</table>',
    'cash_flow': r'consolidated statements of cash flows.*?<table.*?>(.*?)</table>',
}

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

def _get_cache_path(ticker, plot_type):
    return os.path.join(CACHE_DIR, f'{ticker}_{plot_type}_explanation.json')

def _load_from_cache(ticker, plot_type):
    cache_path = _get_cache_path(ticker, plot_type)
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f).get('explanation')
    return None

def _save_to_cache(ticker, plot_type, explanation):
    cache_path = _get_cache_path(ticker, plot_type)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump({'explanation': explanation}, f, indent=2)

# Initialize LLM (ensure GOOGLE_API_KEY is set in your environment)
# api_key = os.getenv('GOOGLE_API_KEY')
api_key = 'AIzaSyD8Lnshp4THbp7jYRF9IHFyxtyGAdEBzfA' # Hardcoded API key from user's last edit
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to use LLM features.")
# llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.2)
llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key='AIzaSyD8Lnshp4THbp7jYRF9IHFyxtyGAdEBzfA', temperature=0, timeout=60)


def _get_llm_explanation(prompt_template_string, data_context, ticker, plot_type):
    """Helper function to get explanation from LLM, with caching."""
    # Try to load from cache first
    cached_explanation = _load_from_cache(ticker, plot_type)
    if cached_explanation:
        print(f"[CACHE HIT] Loaded {plot_type} explanation for {ticker} from cache.")
        return cached_explanation

    # If not in cache, generate with LLM
    print(f"[CACHE MISS] Generating {plot_type} explanation for {ticker} with LLM...")
    prompt = ChatPromptTemplate.from_template(prompt_template_string)
    chain = prompt | llm_model # Use the globally initialized llm_model

    try:
        raw_response = chain.invoke(data_context)
        explanation = ""
        if isinstance(raw_response, AIMessage):
            explanation = raw_response.content
        elif isinstance(raw_response, dict) and 'text' in raw_response:
            explanation = raw_response['text']
        else:
            explanation = str(raw_response) # Fallback for unexpected types

        # Save to cache
        _save_to_cache(ticker, plot_type, explanation)
        print(f"[CACHE SAVE] Saved {plot_type} explanation for {ticker} to cache.")
        return explanation
    except Exception as e:
        print(f"[LLM ERROR] Failed to get explanation for {ticker} {plot_type}: {e}")
        return f"Could not generate explanation due to an error: {e}"

def download_stock_data(ticker, period='6mo'): # Reduced to 6 months
    """Download historical stock data for the given ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def plot_closing_and_moving_averages(df, ticker, output_path):
    """Plot closing price and moving averages for the ticker."""
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    plt.figure(figsize=(10,6)) # Reduced figsize
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['MA50'], label='50-Day MA')
    plt.plot(df['MA200'], label='200-Day MA')
    plt.title(f'{ticker} Closing Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=75) # Reduced DPI
    plt.close()

def calculate_and_plot_returns(df, ticker, output_path):
    """Calculate and plot daily returns for the ticker."""
    df['Daily Return'] = df['Close'].pct_change()
    plt.figure(figsize=(10,5)) # Reduced figsize
    plt.plot(df['Daily Return'], label='Daily Return')
    plt.title(f'{ticker} Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=75) # Reduced DPI
    plt.close()
    return df

def calculate_and_plot_rolling_volatility(df, ticker, output_path, window=21):
    """Calculate and plot rolling volatility for the ticker."""
    df['Rolling Volatility'] = df['Daily Return'].rolling(window=window).std()
    plt.figure(figsize=(10,5)) # Reduced figsize
    plt.plot(df['Rolling Volatility'], label=f'{window}-Day Rolling Volatility')
    plt.title(f'{ticker} Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=75) # Reduced DPI
    plt.close()
    return df

def simulate_buy_and_hold(df, initial_investment=10000):
    """Simulate a buy-and-hold portfolio for the ticker."""
    df['Portfolio Value'] = initial_investment * (1 + df['Daily Return']).cumprod()
    return df

def plot_buy_and_hold(df, ticker, output_path):
    """Plot buy-and-hold portfolio value for the ticker."""
    plt.figure(figsize=(10,5)) # Reduced figsize
    plt.plot(df['Portfolio Value'], label='Buy-and-Hold Portfolio Value')
    plt.title(f'{ticker} Buy-and-Hold Portfolio Simulation')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=75) # Reduced DPI
    plt.close()

def simulate_dca(df, investment_per_period=500, period_days=21):
    """Simulate dollar-cost averaging (DCA) for the ticker."""
    periods = np.arange(0, len(df), period_days)
    shares_accumulated = 0
    portfolio_values = []
    for i in range(len(df)):
        if i in periods:
            shares_accumulated += investment_per_period / df['Close'].iloc[i]
        portfolio_values.append(shares_accumulated * df['Close'].iloc[i])
    df['DCA Portfolio Value'] = portfolio_values
    return df

def plot_dca(df, ticker, output_path):
    """Plot DCA portfolio value for the ticker."""
    plt.figure(figsize=(10,5)) # Reduced figsize
    plt.plot(df['DCA Portfolio Value'], label='DCA Portfolio Value')
    plt.title(f'{ticker} Dollar-Cost Averaging Simulation')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=75) # Reduced DPI
    plt.close()

def simulate_diversified_portfolio(tickers, period='6mo', initial_investment=10000): # Reduced to 6 months
    """Simulate a diversified portfolio including the given ticker, AAPL and SPY."""
    data = yf.download(tickers, period=period)['Close']
    returns = data.pct_change().dropna()
    weights = [1/len(tickers)] * len(tickers)
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_value = initial_investment * (1 + portfolio_returns).cumprod()
    ticker_value = initial_investment * (1 + returns[tickers[0]]).cumprod()
    return portfolio_value, ticker_value

def plot_diversified_portfolio(portfolio_value, ticker_value, ticker, output_path):
    """Plot diversified portfolio vs. single ticker portfolio."""
    plt.figure(figsize=(10,5)) # Reduced figsize
    plt.plot(portfolio_value, label='Diversified Portfolio')
    plt.plot(ticker_value, label=f'{ticker} Only')
    plt.title(f'Diversified Portfolio vs. {ticker} Only')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=75) # Reduced DPI
    plt.close()

#############################################

def setup_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; TestBot/0.1; +spoo@test.ai)'
    })
    return session

def download_file_with_retry(session, url):
    try:
        response = session.get(url) 
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        print(f"Failed to download {url}: {e}")
        return None 

def download_and_unzip_forms(session, start_year, end_year, base_url, target_directory, requests_per_minute=10):
    request_interval = 60 / requests_per_minute 
    for year in range(start_year, end_year + 1):
        for quarter in ['QTR1', 'QTR2', 'QTR3', 'QTR4']:
            zip_url = f"{base_url}/{year}/{quarter}/form.zip"  
            response = download_file_with_retry(session, zip_url)
            if response: 
                try:
                    with zipfile.ZipFile(BytesIO(response.content)) as zfile:
                        extracted_files = zfile.namelist()

                        if any(file.endswith('.idx') for file in extracted_files): 
                            extract_path = os.path.join(target_directory, f"{year}{quarter}")
                            os.makedirs(extract_path, exist_ok=True)  
                            zfile.extractall(path=extract_path)  
                            print(f"Extracted to {extract_path}")
                            extracted_idx_files = glob.glob(os.path.join(extract_path, '*.idx'))
                            if extracted_idx_files:
                                original_file = extracted_idx_files[0]
                                new_filename = os.path.join(target_directory, f"{year}{quarter}.idx")
                                os.rename(original_file, new_filename)
                                print(f"Renamed extracted file to {new_filename}")
                                shutil.rmtree(extract_path)
                                print(f"Removed directory: {extract_path}")
                            else:
                                print("No .idx file found to rename.")
                        else:
                            print(f"No .idx files found in {zip_url}. Skipping extraction.")
                except zipfile.BadZipFile as e:
                    print(f"Failed to unzip {zip_url}: {e}") 
            time.sleep(request_interval)

def parse_idx_file(file_path, form_types, cik):
    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data_start = next(i for i, line in enumerate(lines) if re.match(r'-{3,}', line)) + 1
    for line in lines[data_start:]:
        if True:
            try:
                parts = re.split(r'\s{2,}', line.strip())
            except Exception as e:
                print(f"Error processing line... Error: {e}")
                continue
            if len(parts) == 5:
                form_type, company_name, cik_in, date_filed, file_name = parts

                if form_type in form_types and cik_in in cik:
                    records.append([form_type, company_name, cik.zfill(10), date_filed, file_name])
                    return records
    return records 

def extract_10k_urls(file_path):
    urls = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('10-K'):
                parts = line.split(",")
                url = parts[-1].strip()
                urls.append(f"https://www.sec.gov/Archives/{url}")
    return urls

def download_EDGAR(ticker, cik, root_path, target_directory):
    session = setup_session()  
    start_year = 2025 
    end_year = 2025 
    base_url = "https://www.sec.gov/Archives/edgar/full-index/" 
    requests_per_minute = 10 
    os.makedirs(target_directory, exist_ok=True) 
    
    # download_and_unzip_forms(session, start_year, end_year, base_url, target_directory, requests_per_minute)
    form_types = ['10-K']
    cik_filled = str(cik).zfill(10)
    all_records = []
    for file_name in os.listdir(target_directory):
        if file_name.endswith('.idx'):
            file_path = os.path.join(target_directory, file_name)
            print(f"Parsing file: {file_path}")
            all_records.extend(parse_idx_file(file_path, form_types, cik_filled))

    accumulated_df = pd.DataFrame(all_records, columns=['Form_Type', 'Company_Name', 'CIK', 'Date_Filed', 'File_Name'])
    
    output_file = root_path+"/static/combined_filtered.csv" 
    print('created csv')
    accumulated_df.to_csv(output_file, index=False)
    del all_records
    del accumulated_df
    url_10k = extract_10k_urls(output_file) 
    print(url_10k)

    folder_name = root_path+"/static/10K" 
    shutil.rmtree(folder_name, ignore_errors=True)
    os.makedirs(folder_name, exist_ok=True) 
    headers = {
        'User-Agent': 'Rutgers rv559@rutgers.edu' 
    }        
    for i, url in enumerate(url_10k, start=1):
        start_time = time.time()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            url_path = url.split('/')
            file_name = url_path[-1] 
            
            file_path = os.path.join(folder_name, file_name)
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {file_path}")
        else:
            print(f"Failed to download file from {url}. Status Code: {response.status_code}")
        elapsed_time = time.time() - start_time
        if elapsed_time < requests_per_minute:
            time.sleep(requests_per_minute - elapsed_time)

def concat_csvs(file_list):
    dfs = []
    for f in file_list:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Could not read {f}: {e}")
    return pd.concat(dfs, ignore_index=True)

def analyse_EDGAR(ticker, cik, root_path):
    years = [ '25']

    folder_name = root_path+"/static/10K" 
    folder=root_path+'/static/output-csv'
    out_folder=root_path+"/static/output-txt/"

    extracted_csv_paths = []
    full_html_content = ""

    for input_file in os.listdir(folder_name):
        if True:
            print(folder_name+'/'+input_file)
            year = input_file.split('-')[1]

            #######new method of extracting here###########
            try:
                with open(folder_name+'/'+input_file, "r", encoding="utf-8", errors="ignore") as f:
                    full_html_content = f.read()
                    print(f"DEBUG: Read {len(full_html_content)} bytes for {ticker} {year}. Memory use for raw string: {len(full_html_content) / (1024*1024):.2f} MB")

            except FileNotFoundError:
                print(f"Error: 10-K file not found at {file_path}")
                return []
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return []

            for section_name, headers in section_headers.items(): # Use your existing section_headers for text matching
                section_html_snippet = None
                combined_header_pattern = '|'.join(re.escape(h) for h in headers) 
                pattern = rf'({combined_header_pattern}).*?<table.*?>(.*?)</table>'
                match = re.search(pattern, full_html_content, re.IGNORECASE | re.DOTALL | re.S)
                print('match')
                print(match)

                if match:
                    section_html_snippet = match.group(0) # Get the full matched HTML string including header and table
                    print(f"DEBUG: Found {section_name} snippet (length: {len(section_html_snippet)}).")
                else:
                    print(f"Warning: {section_name} pattern not found in {ticker} {year}. Skipping section.")
                    continue

                df = pd.DataFrame()
                try:
                    if section_html_snippet:
                        # Create a temporary BeautifulSoup object for the snippet
                        section_soup = BeautifulSoup(section_html_snippet, 'lxml') 
                        
                        tables_in_snippet = section_soup.find_all('table')

                        if tables_in_snippet: # Check if list is not empty
                            df = pd.read_html(StringIO(str(tables_in_snippet[0])), flavor='lxml')[0] # Parse the first table found in snippet
                            
                            # Clean up DataFrame (common issue with EDGAR tables)
                            df.dropna(how='all', axis=1, inplace=True) # Drop columns that are all NaN
                            df.dropna(how='all', axis=0, inplace=True) # Drop rows that are all NaN
                            
                            if not df.empty:
                                csv_file_name = f"{ticker}_{year}_{section_name}.csv"
                                csv_file_path = os.path.join(folder, csv_file_name)
                                os.makedirs(folder, exist_ok=True) # Ensure output dir exists
                                df.to_csv(csv_file_path, index=False)
                                print(f"Extracted and saved {section_name} for {ticker} {year} to {csv_file_path}")
                                extracted_csv_paths.append(csv_file_path)
                            else:
                                print(f"Warning: {section_name} table was empty after cleaning for {ticker} {year}.")
                        else:
                            print(f"Warning: BeautifulSoup found no tables in the HTML snippet for {section_name} of {ticker} {year}. Snippet length: {len(section_html_snippet)}")
                    else:
                        print(f"Warning: No HTML snippet extracted for {section_name} for {ticker} {year}.")
                except ValueError as ve: # pd.read_html specific errors
                    print(f"Error parsing HTML table for {section_name} of {ticker} {year}: {ve}. Snippet starts: {section_html_snippet[:500] if section_html_snippet else 'N/A'}...")
                except Exception as e:
                    print(f"An unexpected error occurred processing {section_name} for {ticker} {year}: {e}. Snippet starts: {section_html_snippet[:500] if section_html_snippet else 'N/A'}...")
                
                # Explicitly clear variables for this section to free memory
                del section_html_snippet
                if 'section_soup' in locals(): del section_soup # Delete the temporary soup object
                del df # If df is no longer needed after saving to CSV

            # Explicitly clear the full HTML content after all sections are processed for the current file
            del full_html_content
                        

################old method of extracting here#####################

                # found = False
                # for tag in soup.find_all(text=True):
                #     if any(header in tag.lower() for header in headers):
                #         # Find the next table after the header
                #         next_table = tag.find_parent().find_next('table')
                #         if next_table:
                #             df = pd.read_html(StringIO(str(next_table)))[0]
                #             # shutil.rmtree(folder, ignore_errors=True)
                #             os.makedirs(folder, exist_ok=True) 
                #             csv_file = folder+f"/{ticker}_{year}_{section}.csv"
                #             df.to_csv(csv_file, index=False)
                #             del df
                #             print(f"Saved {section} to {csv_file}")
                #             found = True
                #             break
                # if not found:
                #     print(f"{section} not found in {input_file}")

    balance_sheets = [f"{folder}/{ticker}_{y}_balance_sheet.csv" for y in years]
    cash_flows = [f"{folder}/{ticker}_{y}_cash_flow.csv" for y in years]
    income_statements = [f"{folder}/{ticker}_{y}_income_statement.csv" for y in years]
    print('concatenating')

    df_balance = concat_csvs(balance_sheets)
    balance_text = df_balance.to_string(index=False)
    print('balance sheet done')
    del df_balance
    
    df_cash = concat_csvs(cash_flows)
    cash_text = df_cash.to_string(index=False)
    print('cashflow done')
    del df_cash

    df_income = concat_csvs(income_statements)
    income_text = df_income.to_string(index=False)
    print('income statement done')
    del df_income
    

    prompt_template = (
        """
        You are a financial analyst. Given the following financial statement tables for a company, perform a comprehensive analysis as follows:

        Here are the financial statements:

        Balance Sheet:
        {balance_text}

        Cash Flow Statement:
        {cash_text}

        Income Statement:
        {income_text}

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

        Please structure your response in three sections only and keep the response professional to the user without personal expressions. Use the available data only and do not expect more data for now:
        1. Financial Trends Analysis
        2. Insights and Explanations
        3. Valuation Advice
        """
    )

    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyD8Lnshp4THbp7jYRF9IHFyxtyGAdEBzfA", temperature=0, timeout=60)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm_model

    result = chain.invoke({
        "balance_text": balance_text,
        "cash_text": cash_text,
        "income_text": income_text
    })
    print('AI result')
    print(result)

    # Save output
    shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder)
    with open(out_folder+f"{ticker}-LLM-analysis.txt", "w", encoding="utf-8") as out:
        out.write(result.content)
    return result.content



def run_full_analysis(ticker, cik, root_path, output_dir='static/plots'):
    INDEX_FILES_DIR = os.path.join(root_path, 'static', 'index_files')

    os.makedirs(output_dir, exist_ok=True)
    try:
        download_EDGAR(ticker, cik, root_path, INDEX_FILES_DIR)
        analysis_text = analyse_EDGAR(ticker, cik, root_path)
    except Exception as e:
        print('Error while downloading and analysing')
        analysis_text = 'Error'
    df = download_stock_data(ticker)
    plots_info = []

    if df.empty:
        return [(f'{ticker}_error.png', f'Could not retrieve data for {ticker}. Please check the ticker symbol or data availability.')]

    # 1. Closing price and moving averages
    closing_ma_path = os.path.join(output_dir, f'{ticker}_closing_ma.png')
    plot_closing_and_moving_averages(df, ticker, closing_ma_path)
    plots_info.append((f'{ticker}_closing_ma.png', 'Shows price trend and moving averages.'))
    # closing_ma_explanation_prompt = (
    #     """
    #     Explain the trends in the closing stock price for {ticker} based on the following data:
    #     Closing Price Summary (last 5 years):
    #     {close_summary}
        
    #     What do these trends suggest about the stock's short-term and long-term momentum?
    #     Keep it concise, around 2-3 sentences.
    #     """
    # )
    # # Removed hardcoded API key from LLM init, passed from run_full_analysis via _get_llm_explanation
    # closing_ma_explanation = _get_llm_explanation(
    #     closing_ma_explanation_prompt,
    #     {
    #         'ticker': ticker,
    #         'close_summary': df['Close'].to_string(index=False)
    #         # 'ma50_summary': df['MA50'].describe().to_string(),
    #         # 'ma200_summary': df['MA200'].describe().to_string()
    #     },
    #     ticker,
    #     'closing_ma'
    # )
    # plots_info.append((f'{ticker}_closing_ma.png', closing_ma_explanation))

    # 2. Daily returns
    returns_path = os.path.join(output_dir, f'{ticker}_daily_returns.png')
    df = calculate_and_plot_returns(df, ticker, returns_path)
    plots_info.append((f'{ticker}_daily_returns.png', 'Shows daily return volatility.'))
    # daily_returns_explanation_prompt = (
    #     """
    #     Explain the daily return distribution and volatility for {ticker} based on the following data:
    #     Daily Return Summary:
    #     {daily_return_summary}
    #     What does the mean, standard deviation, min, and max daily return tell about the stock's performance and risk?
    #     Keep it concise, around 2-3 sentences.
    #     """
    # )
    # daily_returns_explanation = _get_llm_explanation(
    #     daily_returns_explanation_prompt,
    #     {
    #         'ticker': ticker,
    #         'daily_return_summary': df['Daily Return'].describe().to_string()
    #     },
    #     ticker,
    #     'daily_returns'
    # )
    # plots_info.append((f'{ticker}_daily_returns.png', daily_returns_explanation))

    # 3. Rolling volatility
    volatility_path = os.path.join(output_dir, f'{ticker}_rolling_volatility.png')
    df = calculate_and_plot_rolling_volatility(df, ticker, volatility_path)
    plots_info.append((f'{ticker}_rolling_volatility.png', 'Shows rolling volatility (risk) over time.'))
    # rolling_volatility_explanation_prompt = (
    #     """
    #     Explain the trends in the rolling volatility for {ticker} based on the following data:
    #     Rolling Volatility Summary:
    #     {rolling_volatility_summary}
    #     What does the change in volatility over time suggest about the stock's risk profile?
    #     Keep it concise, around 2-3 sentences.
    #     """
    # )
    # rolling_volatility_explanation = _get_llm_explanation(
    #     rolling_volatility_explanation_prompt,
    #     {
    #         'ticker': ticker,
    #         'rolling_volatility_summary': df['Rolling Volatility'].describe().to_string()
    #     },
    #     ticker,
    #     'rolling_volatility'
    # )
    # plots_info.append((f'{ticker}_rolling_volatility.png', rolling_volatility_explanation))

    # 4. Buy-and-hold simulation
    df = simulate_buy_and_hold(df)
    buy_hold_path = os.path.join(output_dir, f'{ticker}_buy_and_hold.png')
    plot_buy_and_hold(df, ticker, buy_hold_path)
    plots_info.append((f'{ticker}_buy_and_hold.png', 'Simulates a buy-and-hold investment strategy.'))
    # buy_hold_explanation_prompt = (
    #     """
    #     Explain the outcome of a buy-and-hold investment for {ticker} based on the following data:
    #     Initial Investment: $10000
    #     Final Portfolio Value: ${final_value:.2f}
    #     What does this simulation indicate about the stock's long-term performance?
    #     Keep it concise, around 2-3 sentences.
    #     """
    # )
    # buy_hold_explanation = _get_llm_explanation(
    #     buy_hold_explanation_prompt,
    #     {
    #         'ticker': ticker,
    #         'final_value': df['Portfolio Value'].iloc[-1]
    #     },
    #     ticker,
    #     'buy_and_hold'
    # )
    # plots_info.append((f'{ticker}_buy_and_hold.png', buy_hold_explanation))

    # 5. Dollar-cost averaging simulation
    df = simulate_dca(df)
    dca_path = os.path.join(output_dir, f'{ticker}_dca.png')
    plot_dca(df, ticker, dca_path)
    plots_info.append((f'{ticker}_dca.png', 'Simulates a dollar-cost averaging investment strategy.'))
    # dca_explanation_prompt = (
    #     """
    #     Explain the outcome of a dollar-cost averaging (DCA) investment for {ticker} based on the following data:
    #     Initial Investment (implied total from $500/month over 5 years)
    #     Final Portfolio Value: ${final_value:.2f}
    #     How does this compare to a simple buy-and-hold strategy, and what does it suggest about mitigating volatility?
    #     Keep it concise, around 2-3 sentences.
    #     """
    # )
    # dca_explanation = _get_llm_explanation(
    #     dca_explanation_prompt,
    #     {
    #         'ticker': ticker,
    #         'final_value': df['DCA Portfolio Value'].iloc[-1]
    #     },
    #     ticker,
    #     'dca'
    # )
    # plots_info.append((f'{ticker}_dca.png', dca_explanation))

    # 6. Diversified portfolio simulation
    tickers = [ticker, 'AAPL' if ticker!='AAPL' else 'NVDA', 'FDX' if ticker!='FDX' else 'HD']
    portfolio_value, ticker_value = simulate_diversified_portfolio(tickers)
    diversified_path = os.path.join(output_dir, f'{ticker}_diversified_portfolio.png')
    # plot_diversified_portfolio(portfolio_value, ticker_value, ticker, diversified_path)
    # plots_info.append((f'{ticker}_diversified_portfolio.png', f'Compares a diversified portfolio ({ticker}, {tickers[1]}, {tickers[2]}) to {ticker} alone.'))
    # diversified_explanation_prompt = (
    #     """
    #     Explain the comparison between a diversified portfolio ({ticker}, {ticker2}, {ticker3}) and investing solely in {ticker} based on the following data:
    #     Diversified Portfolio Final Value: ${diversified_final_value:.2f}
    #     {ticker} Only Final Value: ${ticker_only_final_value:.2f}
    #     What does this comparison suggest about the benefits of diversification for {ticker}?
    #     Keep it concise, around 2-3 sentences.
    #     """
    # )
    # diversified_explanation = _get_llm_explanation(
    #     diversified_explanation_prompt,
    #     {
    #         'ticker': ticker,
    #         'ticker2': diversified_tickers[1],
    #         'ticker3': diversified_tickers[2],
    #         'diversified_final_value': portfolio_value.iloc[-1],
    #         'ticker_only_final_value': ticker_value.iloc[-1]
    #     },
    #     ticker,
    #     'diversified_portfolio'
    # )
    # plots_info.append((f'{ticker}_diversified_portfolio.png', diversified_explanation))
    plots_info.append(('analysis', analysis_text))

    return plots_info 