from flask import Flask, jsonify, send_from_directory, request
import pandas as pd
import os
from backend_analysis import run_full_analysis

app = Flask(__name__, static_folder='static')

# Load S&P 500 companies
sp500 = pd.read_csv('sp100_list.csv')  # columns: Symbol, Name

@app.route('/api/companies')
def get_companies():
    companies = sp500[['tic', 'conm']].to_dict(orient='records')
    return jsonify(companies)

@app.route('/api/analysis/<ticker>')
def get_analysis(ticker):
    print(ticker)
    # Here, call your analysis functions (from your previous script)
    plot_info = run_full_analysis(ticker)

    plots = [
        {
            'title': os.path.splitext(filename)[0].replace(f'{ticker}_', '').replace('_', ' ').title(),
            'url': f'/static/plots/{filename}',
            'explanation': explanation
        }
        for filename, explanation in plot_info
    ]
    return jsonify({'plots': plots})

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    return send_from_directory('static/plots', filename)

if __name__ == '__main__':
    app.run(debug=True)