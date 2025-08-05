import os
import yfinance as yf
import matplotlib
matplotlib.use('Agg') # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def run_full_analysis(ticker, output_dir='static/plots'):
    """
    Run all analysis and plotting functions for the given ticker.
    Saves plots as {ticker}_plotname.png in output_dir.
    Returns a list of (filename, explanation) tuples.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = download_stock_data(ticker)
    plots_info = []

    # 1. Closing price and moving averages
    closing_ma_path = os.path.join(output_dir, f'{ticker}_closing_ma.png')
    print(closing_ma_path)
    plot_closing_and_moving_averages(df, ticker, closing_ma_path)
    plots_info.append((f'{ticker}_closing_ma.png', 'Shows price trend and moving averages.'))
    print(plots_info)

    # 2. Daily returns
    returns_path = os.path.join(output_dir, f'{ticker}_daily_returns.png')
    df = calculate_and_plot_returns(df, ticker, returns_path)
    plots_info.append((f'{ticker}_daily_returns.png', 'Shows daily return volatility.'))

    # 3. Rolling volatility
    volatility_path = os.path.join(output_dir, f'{ticker}_rolling_volatility.png')
    df = calculate_and_plot_rolling_volatility(df, ticker, volatility_path)
    plots_info.append((f'{ticker}_rolling_volatility.png', 'Shows rolling volatility (risk) over time.'))

    # 4. Buy-and-hold simulation
    df = simulate_buy_and_hold(df)
    buy_hold_path = os.path.join(output_dir, f'{ticker}_buy_and_hold.png')
    plot_buy_and_hold(df, ticker, buy_hold_path)
    plots_info.append((f'{ticker}_buy_and_hold.png', 'Simulates a buy-and-hold investment strategy.'))

    # 5. Dollar-cost averaging simulation
    df = simulate_dca(df)
    dca_path = os.path.join(output_dir, f'{ticker}_dca.png')
    plot_dca(df, ticker, dca_path)
    plots_info.append((f'{ticker}_dca.png', 'Simulates a dollar-cost averaging investment strategy.'))

    # 6. Diversified portfolio simulation
    tickers = [ticker, 'AAPL' if ticker!='AAPL' else 'NVDA', 'FDX' if ticker!='FDX' else 'HD']
    portfolio_value, ticker_value, _ = simulate_diversified_portfolio(tickers)
    diversified_path = os.path.join(output_dir, f'{ticker}_diversified_portfolio.png')
    plot_diversified_portfolio(portfolio_value, ticker_value, ticker, diversified_path)
    plots_info.append((f'{ticker}_diversified_portfolio.png', f'Compares a diversified portfolio ({ticker}, {tickers[1]}, {tickers[2]}) to {ticker} alone.'))

    return plots_info 