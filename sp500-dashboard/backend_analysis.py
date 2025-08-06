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

CACHE_DIR = 'llm_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key='AIzaSyD8Lnshp4THbp7jYRF9IHFyxtyGAdEBzfA', temperature=0)


def _get_llm_explanation(prompt_template_string, data_context, ticker, plot_type):
    """Helper function to get explanation from LLM, with caching."""

    cached_explanation = _load_from_cache(ticker, plot_type)
    if cached_explanation:
        print(f"[CACHE HIT] Loaded {plot_type} explanation for {ticker} from cache.")
        return cached_explanation

    print(f"[CACHE MISS] Generating {plot_type} explanation for {ticker} with LLM...")
    prompt = ChatPromptTemplate.from_template(prompt_template_string)
    print(prompt)
    chain = prompt | llm
    response = chain.invoke(data_context)
    explanation = response['text'] if 'text' in response else str(response)

    _save_to_cache(ticker, plot_type, explanation)
    print(f"[CACHE SAVE] Saved {plot_type} explanation for {ticker} to cache.")
    return explanation

def download_stock_data(ticker, period='5y'):
    """Download historical stock data for the given ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def plot_closing_and_moving_averages(df, ticker, output_path):
    """Plot closing price and moving averages for the ticker."""
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['MA50'], label='50-Day MA')
    plt.plot(df['MA200'], label='200-Day MA')
    plt.title(f'{ticker} Closing Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def calculate_and_plot_returns(df, ticker, output_path):
    """Calculate and plot daily returns for the ticker."""
    df['Daily Return'] = df['Close'].pct_change()
    plt.figure(figsize=(14,5))
    plt.plot(df['Daily Return'], label='Daily Return')
    plt.title(f'{ticker} Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return df

def calculate_and_plot_rolling_volatility(df, ticker, output_path, window=21):
    """Calculate and plot rolling volatility for the ticker."""
    df['Rolling Volatility'] = df['Daily Return'].rolling(window=window).std()
    plt.figure(figsize=(14,5))
    plt.plot(df['Rolling Volatility'], label=f'{window}-Day Rolling Volatility')
    plt.title(f'{ticker} Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return df

def simulate_buy_and_hold(df, initial_investment=10000):
    """Simulate a buy-and-hold portfolio for the ticker."""
    df['Portfolio Value'] = initial_investment * (1 + df['Daily Return']).cumprod()
    return df

def plot_buy_and_hold(df, ticker, output_path):
    """Plot buy-and-hold portfolio value for the ticker."""
    plt.figure(figsize=(14,5))
    plt.plot(df['Portfolio Value'], label='Buy-and-Hold Portfolio Value')
    plt.title(f'{ticker} Buy-and-Hold Portfolio Simulation')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
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
    plt.figure(figsize=(14,5))
    plt.plot(df['DCA Portfolio Value'], label='DCA Portfolio Value')
    plt.title(f'{ticker} Dollar-Cost Averaging Simulation')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def simulate_diversified_portfolio(tickers, period='5y', initial_investment=10000):
    """Simulate a diversified portfolio including the given ticker, AAPL and SPY."""
    data = yf.download(tickers, period=period)['Close']
    returns = data.pct_change().dropna()
    weights = [1/len(tickers)] * len(tickers)
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_value = initial_investment * (1 + portfolio_returns).cumprod()
    ticker_value = initial_investment * (1 + returns[tickers[0]]).cumprod()
    return portfolio_value, ticker_value, data

def plot_diversified_portfolio(portfolio_value, ticker_value, ticker, output_path):
    """Plot diversified portfolio vs. single ticker portfolio."""
    plt.figure(figsize=(14,5))
    plt.plot(portfolio_value, label='Diversified Portfolio')
    plt.plot(ticker_value, label=f'{ticker} Only')
    plt.title(f'Diversified Portfolio vs. {ticker} Only')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
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
    print('cik')
    print(cik)
    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data_start = next(i for i, line in enumerate(lines) if re.match(r'-{3,}', line)) + 1
    for line in lines[data_start:]:
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) == 5:
            form_type, company_name, cik_in, date_filed, file_name = parts
            # cik_padded = cik_in.zfill(10)
            # if form_type in form_types:
                # print(f"Found matching form type: {form_type}, CIK: {cik_padded}")
            if (cik==cik_in):
                print('true')
            if form_type in form_types and cik_in in cik:
                records.append([form_type, company_name, cik.zfill(10), date_filed, file_name])
    print(records)
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
    start_year = 2024 
    end_year = 2025 
    base_url = "https://www.sec.gov/Archives/edgar/full-index/" 
    requests_per_minute = 10 
    os.makedirs(target_directory, exist_ok=True) 
    
    download_and_unzip_forms(session, start_year, end_year, base_url, target_directory, requests_per_minute)
    form_types = ['10-K']
    cik_filled = str(cik).zfill(10)
    all_records = []
    for file_name in os.listdir(target_directory):
        if file_name.endswith('.idx'):
            file_path = os.path.join(target_directory, file_name)
            print(f"Parsing file: {file_path}")
            all_records.extend(parse_idx_file(file_path, form_types, cik_filled))

    print('reached')
    print(all_records)
    accumulated_df = pd.DataFrame(all_records, columns=['Form_Type', 'Company_Name', 'CIK', 'Date_Filed', 'File_Name'])
    output_file = root_path+"/static/combined_filtered.csv" 
    accumulated_df.to_csv(output_file, index=False)
    url_10k = extract_10k_urls(output_file) 
    print('urls')
    print(url_10k)
    folder_name = root_path+"/static/10K" 
    shutil.rmtree(folder_name, ignore_errors=True)
    os.makedirs(folder_name, exist_ok=True) 
    headers = {
        'User-Agent': 'TestBot +spoo@test.ai' 
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
    years = ['22','23', '24', '25']

    folder_name = root_path+"/static/10K" 
    folder=root_path+'/static/output-csv'
    out_folder=root_path+"/static/output-txt/"
    for input_file in os.listdir(folder_name):
        if True:
            print(folder_name+'/'+input_file)
            year = input_file.split('-')[1]
            with open(folder_name+'/'+input_file, "r", encoding="utf-8", errors="ignore") as f:
                print('inside')
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
                            # shutil.rmtree(folder, ignore_errors=True)
                            os.makedirs(folder, exist_ok=True) 
                            csv_file = folder+f"/{ticker}_{year}_{section}.csv"
                            df.to_csv(csv_file, index=False)
                            print(f"Saved {section} to {csv_file}")
                            found = True
                            break
                if not found:
                    print(f"{section} not found in {input_file}")

    balance_sheets = [f"{folder}/{ticker}_{y}_balance_sheet.csv" for y in years]
    cash_flows = [f"{folder}/{ticker}_{y}_cash_flow.csv" for y in years]
    income_statements = [f"{folder}/{ticker}_{y}_income_statement.csv" for y in years]
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

        Please structure your response in three sections only and do not return anything extra:
        1. Financial Trends Analysis
        2. Insights and Explanations
        3. Valuation Advice
        """
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyD8Lnshp4THbp7jYRF9IHFyxtyGAdEBzfA", temperature=0)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm

    result = chain.invoke({
        "balance_text": balance_text,
        "cash_text": cash_text,
        "income_text": income_text
    })

    print("\n--- AI Summary ---\n")
    print(result.content)

    # Save output
    shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder)
    with open(out_folder+f"{ticker}-LLM-analysis.txt", "w", encoding="utf-8") as out:
        out.write(result.content)
        print('written')
    return result.content



def run_full_analysis(ticker, cik, root_path, output_dir='static/plots'):
    STATIC_FOLDER_PATH = os.path.join(root_path, 'static')
    INDEX_FILES_DIR = os.path.join(root_path, 'index_files')
    PLOTS_DIR = os.path.join(root_path, 'static', 'plots')

    os.makedirs(output_dir, exist_ok=True)
    download_EDGAR(ticker, cik, root_path, INDEX_FILES_DIR)
    analysis_text = analyse_EDGAR(ticker, cik, root_path)
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
    portfolio_value, ticker_value, _ = simulate_diversified_portfolio(tickers)
    diversified_path = os.path.join(output_dir, f'{ticker}_diversified_portfolio.png')
    plot_diversified_portfolio(portfolio_value, ticker_value, ticker, diversified_path)
    plots_info.append((f'{ticker}_diversified_portfolio.png', f'Compares a diversified portfolio ({ticker}, {tickers[1]}, {tickers[2]}) to {ticker} alone.'))
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