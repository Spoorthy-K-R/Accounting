from flask import Flask, jsonify, send_from_directory, request
import pandas as pd
import os
from backend_analysis import run_full_analysis
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app) 

# Load S&P 500 companies
sp500 = pd.read_csv('sp100_list.csv')

@app.route('/api/companies')
def get_companies():
    companies = sp500[['tic', 'conm']].to_dict(orient='records')
    return jsonify(companies)

@app.route('/api/analysis/<ticker>')
def get_analysis(ticker):
    try:
        cik = sp500.loc[sp500['tic']==ticker, 'cik']
        plot_info = run_full_analysis(ticker, str(cik).split()[1], app.root_path)
        analysis_key, analysis_text = plot_info.pop()
        print(analysis_key)
        print('plot_info')
        print(plot_info)

        plots = [
            {
                'title': os.path.splitext(filename)[0].replace(f'{ticker}_', '').replace('_', ' ').title(),
                'url': f'/static/plots/{filename}',
                'explanation': explanation
            }
            for filename, explanation in plot_info
        ]
        print('json')
        print(jsonify({'plots': plots, 'text': analysis_text}))
        return jsonify({'plots': plots, 'text': analysis_text})
    except Exception as e:
        app.logger.error(f"Error processing analysis for {ticker}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    return send_from_directory('static/plots', filename)

if __name__ == '__main__':
    app.run(debug=True)