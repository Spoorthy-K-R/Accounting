import os
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = 'output_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_nvda_data(period='5y'):
    nvda = yf.Ticker('NVDA')
    df = nvda.history(period=period)
    return df

def plot_closing_and_moving_averages(df, output_path):
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['MA50'], label='50-Day MA')
    plt.plot(df['MA200'], label='200-Day MA')
    plt.title('NVIDIA Closing Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def calculate_and_plot_returns(df, output_path):
    df['Daily Return'] = df['Close'].pct_change()
    plt.figure(figsize=(14,5))
    plt.plot(df['Daily Return'], label='Daily Return')
    plt.title('NVIDIA Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return df

def calculate_and_plot_rolling_volatility(df, output_path, window=21):
    df['Rolling Volatility'] = df['Daily Return'].rolling(window=window).std()
    plt.figure(figsize=(14,5))
    plt.plot(df['Rolling Volatility'], label=f'{window}-Day Rolling Volatility')
    plt.title('NVIDIA Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return df

def simulate_buy_and_hold(df, initial_investment=10000):
    df['Portfolio Value'] = initial_investment * (1 + df['Daily Return']).cumprod()
    return df

def plot_buy_and_hold(df, output_path):
    plt.figure(figsize=(14,5))
    plt.plot(df['Portfolio Value'], label='Buy-and-Hold Portfolio Value')
    plt.title('NVIDIA Buy-and-Hold Portfolio Simulation')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def simulate_dca(df, investment_per_period=500, period_days=21):
    periods = np.arange(0, len(df), period_days)
    shares_accumulated = 0
    portfolio_values = []
    for i in range(len(df)):
        if i in periods:
            shares_accumulated += investment_per_period / df['Close'].iloc[i]
        portfolio_values.append(shares_accumulated * df['Close'].iloc[i])
    df['DCA Portfolio Value'] = portfolio_values
    return df

def plot_dca(df, output_path):
    plt.figure(figsize=(14,5))
    plt.plot(df['DCA Portfolio Value'], label='DCA Portfolio Value')
    plt.title('NVIDIA Dollar-Cost Averaging Simulation')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def simulate_diversified_portfolio(tickers=['NVDA', 'AAPL', 'SPY'], period='5y', initial_investment=10000):
    data = yf.download(tickers, period=period)['Close']
    returns = data.pct_change().dropna()
    weights = [1/len(tickers)] * len(tickers)
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_value = initial_investment * (1 + portfolio_returns).cumprod()
    nvda_value = initial_investment * (1 + returns['NVDA']).cumprod()
    return portfolio_value, nvda_value, data

def plot_diversified_portfolio(portfolio_value, nvda_value, output_path):
    plt.figure(figsize=(14,5))
    plt.plot(portfolio_value, label='Diversified Portfolio')
    plt.plot(nvda_value, label='NVIDIA Only')
    plt.title('Diversified Portfolio vs. NVIDIA Only')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def print_summary(df, portfolio_value, nvda_value):
    print('--- NVIDIA Stock Analysis Summary ---')
    print('Price stats (last 5 years):')
    print(df['Close'].describe())
    print('\nBuy-and-Hold Final Value: ${:.2f}'.format(df['Portfolio Value'].iloc[-1]))
    print('DCA Final Value: ${:.2f}'.format(df['DCA Portfolio Value'].iloc[-1]))
    print('Diversified Portfolio Final Value: ${:.2f}'.format(portfolio_value.iloc[-1]))
    print('NVIDIA Only Final Value: ${:.2f}'.format(nvda_value.iloc[-1]))
    print('\nVolatility stats:')
    print(df['Rolling Volatility'].describe())
    print('-------------------------------------')

def main():
    df = download_nvda_data()
    plot_closing_and_moving_averages(df, os.path.join(OUTPUT_DIR, 'nvda_closing_ma.png'))
    df = calculate_and_plot_returns(df, os.path.join(OUTPUT_DIR, 'nvda_daily_returns.png'))
    df = calculate_and_plot_rolling_volatility(df, os.path.join(OUTPUT_DIR, 'nvda_rolling_volatility.png'))
    df = simulate_buy_and_hold(df)
    plot_buy_and_hold(df, os.path.join(OUTPUT_DIR, 'nvda_buy_and_hold.png'))
    df = simulate_dca(df)
    plot_dca(df, os.path.join(OUTPUT_DIR, 'nvda_dca.png'))
    portfolio_value, nvda_value, _ = simulate_diversified_portfolio()
    plot_diversified_portfolio(portfolio_value, nvda_value, os.path.join(OUTPUT_DIR, 'nvda_diversified_portfolio.png'))
    print_summary(df, portfolio_value, nvda_value)

if __name__ == '__main__':
    main() 