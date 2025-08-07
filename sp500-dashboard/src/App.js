import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [companies, setCompanies] = useState([]);
  const [selected, setSelected] = useState('');
  const [plots, setPlots] = useState([]);
  const [analysis, setAnalysis] = useState([]);

  const [isLoading, setIsLoading] = useState(false); 
  const [error, setError] = useState(null); 

  const [showDiversifiedInput, setShowDiversifiedInput] = useState(false); 
  const [otherTickersInput, setOtherTickersInput] = useState(''); 
  const [diversifiedPlot, setDiversifiedPlot] = useState(null); 
  const [isDiversifiedLoading, setIsDiversifiedLoading] = useState(false); 
  const [diversifiedError, setDiversifiedError] = useState(null); 


  // const FLASK_BACKEND_URL = 'https://financial-data-analysis.onrender.com';
  const FLASK_BACKEND_URL = 'http://127.0.0.1:5000';


  useEffect(() => {
    axios.get(`${FLASK_BACKEND_URL}/api/companies`)
      .then(res => setCompanies(res.data))
      .catch(err => {
        console.error("Failed to fetch companies list:", err);
        setError("Could not load company list. Please try refreshing.");
      });
  }, []);

  const handleChange = (e) => { 
    const newSelection = e.target.value;
    setSelected(newSelection);
    setPlots([]); 
    setAnalysis('');
    setError(null);
    setDiversifiedPlot(null); 
    setDiversifiedError(null); 
    setShowDiversifiedInput(false);
    
    if (!newSelection) { 
      return; 
    }

    setIsLoading(true);

    axios.get(`${FLASK_BACKEND_URL}/api/analysis/${newSelection}`)
    .then(res => {
      setPlots(res.data.plots)
      setAnalysis(res.data.text)
    })
    .catch(err => {
      console.error("Error fetching analysis:", err);
      setError(`Failed to load analysis for ${newSelection}. Please try again.`);
    })
    .finally(() => {
      setIsLoading(false); // Stop loading regardless of success or failure
    });
  };

  const handleDiversifiedAnalysisSubmit = () => {
    if (!selected) {
      setDiversifiedError("Please select a primary company first.");
      return;
    }
    const otherTickersArray = otherTickersInput.split(',').map(ticker => ticker.trim().toUpperCase()).filter(t => t);
    if (otherTickersArray.length === 0) {
      setDiversifiedError("Please enter 2-3 other ticker symbols.");
      return;
    }
    if (otherTickersArray.length > 3) {
      setDiversifiedError("Please enter a maximum of 3 other ticker symbols.");
      return;
    }

    setIsDiversifiedLoading(true);
    setDiversifiedError(null);
    setDiversifiedPlot(null);

    console.log('otherTickers')
    console.log(otherTickersArray)
    console.log('plots')
    console.log(plots)

    axios.post(`${FLASK_BACKEND_URL}/api/diversified-analysis`, {
      primaryTicker: selected,
      otherTickers: otherTickersArray.join(',') // Send as comma-separated string
    })
    .then(res => {
      setDiversifiedPlot(res.data.plot);
      console.log('diverse plot')
      console.log(diversifiedPlot); // Assuming backend returns a single plot object
    })
    .catch(err => {
      console.error("Error fetching diversified analysis:", err);
      setDiversifiedError(`Failed to load diversified analysis: ${err.response?.data?.error || err.message}`);
    })
    .finally(() => {
      setIsDiversifiedLoading(false);
    });
  };

  const handleAnalysis = (e) => {
    console.log('hi')
    axios.get(`${FLASK_BACKEND_URL}/api/analysis/LLM/${selected}`)
    .then(res => {
      console.log('here atleast')
      setAnalysis(res.data)
    })
  }

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', padding: '20px', maxWidth: '900px', margin: '0 auto', textAlign: 'center' }}>
      <h1>S&P 100 Company Analysis Dashboard</h1>
      
      <div style={{ marginBottom: '30px' }}>
        <select 
          onChange={handleChange} 
          value={selected} 
          style={{ padding: '10px', fontSize: '16px', borderRadius: '5px', border: '1px solid #ccc', minWidth: '250px' }}
        >
          <option value="">Select a company</option>
          {companies.map(c => (
            <option key={c.tic} value={c.tic}>{c.conm} ({c.tic})</option>
          ))}
        </select>
        {/* <button onChange={handleAnalysis}>Display LLM Analysis</button> */}
        {selected && ( // Only show button if a company is selected
          <button 
            onClick={() => setShowDiversifiedInput(!showDiversifiedInput)}
            style={{ 
              padding: '10px 15px', 
              fontSize: '16px', 
              borderRadius: '5px', 
              border: 'none', 
              backgroundColor: '#4CAF50', 
              color: 'white', 
              cursor: 'pointer',
              transition: 'background-color 0.3s ease'
            }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#45a049'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#4CAF50'}
          >
            {showDiversifiedInput ? 'Hide Diversified Input' : 'Diversified Analysis'}
          </button>
        )}
      </div>

      {showDiversifiedInput && selected && (
        <div style={{ 
          border: '1px solid #b3e0ff', 
          borderRadius: '8px', 
          padding: '20px', 
          marginBottom: '30px', 
          backgroundColor: '#e6f7ff', 
          textAlign: 'left',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '15px'
        }}>
          <h3 style={{ color: '#0056b3', marginBottom: '10px' }}>Compare {selected} with:</h3>
          <input
            type="text"
            placeholder="Enter 2-3 other tickers (e.g., AAPL, MSFT)"
            value={otherTickersInput}
            onChange={(e) => setOtherTickersInput(e.target.value)}
            style={{ padding: '10px', fontSize: '16px', borderRadius: '5px', border: '1px solid #ccc', width: '80%' }}
          />
          <button
            onClick={handleDiversifiedAnalysisSubmit}
            disabled={isDiversifiedLoading || !otherTickersInput.trim()} // Disable if loading or input is empty
            style={{ 
              padding: '10px 20px', 
              fontSize: '16px', 
              borderRadius: '5px', 
              border: 'none', 
              backgroundColor: '#007bff', 
              color: 'white', 
              cursor: 'pointer',
              opacity: isDiversifiedLoading || !otherTickersInput.trim() ? 0.6 : 1,
              transition: 'background-color 0.3s ease, opacity 0.3s ease'
            }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = isDiversifiedLoading || !otherTickersInput.trim() ? '#007bff' : '#0056b3'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#007bff'}
          >
            {isDiversifiedLoading ? 'Generating...' : 'Generate Diversified Plot'}
          </button>
          
          {isDiversifiedLoading && <p style={{ color: '#555' }}>Loading diversified analysis...</p>}
          {diversifiedError && <p style={{ color: 'red' }}>Error: {diversifiedError}</p>}
        </div>
      )}

      {error && (
        <div style={{ color: 'red', border: '1px solid red', padding: '10px', borderRadius: '5px', marginBottom: '20px' }}>
          <p>Error: {error}</p>
        </div>
      )}

      {isLoading && (
        <div style={{ padding: '20px', color: '#555' }}>
          <p>Loading analysis for {selected}... This may take a moment due to data fetching and LLM processing.</p>
          {/* You could add a simple spinner GIF here if you want */}
        </div>
      )}

      {!isLoading && !error && plots.length === 0 && analysis=='' && selected && !isDiversifiedLoading && !diversifiedError && (
        <div style={{ padding: '20px', color: '#777' }}>
          <p>No analysis available for {selected} or data is still processing.</p>
        </div>
      )}

      {!isLoading && !error && plots.length === 0 && !selected && !isDiversifiedLoading && !diversifiedError && (
        <div style={{ padding: '20px', color: '#777' }}>
          <p>Select a company from the dropdown to view its stock analysis dashboard.</p>
        </div>
      )}

      {!isLoading && !error && plots.length > 0 && (
        <div style={{ marginTop: '20px' }}>
        <h2>Analysis Results for {selected}</h2>

        {showDiversifiedInput && diversifiedPlot.length>0 && (
            <div style={{ border: '1px solid #ccf', borderRadius: '8px', padding: '15px', marginBottom: '25px', backgroundColor: '#f0f4ff' }}>
              <h3 style={{ color: '#003366', marginBottom: '10px' }}>{diversifiedPlot[0].title}</h3>
              <img src={`${FLASK_BACKEND_URL}${diversifiedPlot[0].url}`} alt={diversifiedPlot[0].title} style={{maxWidth: '100%', height: 'auto', borderRadius: '4px'}} />
              <p style={{ fontSize: '0.95em', color: '#444', marginTop: '15px', lineHeight: '1.4' }}>{diversifiedPlot[0].explanation}</p>
            </div>
        )}

        {plots.map(plot => (
            <div key={plot.title} style={{ border: '1px solid #eee', borderRadius: '8px', padding: '15px', marginBottom: '25px', backgroundColor: '#f9f9f9' }}>
              <h3 style={{ color: '#333', marginBottom: '10px' }}>{plot.title}</h3>
              <img src={`${FLASK_BACKEND_URL}${plot.url}`} alt={plot.title} style={{maxWidth: '100%', height: 'auto', borderRadius: '4px'}} />
              <p style={{ fontSize: '0.95em', color: '#666', marginTop: '15px', lineHeight: '1.4' }}>{plot.explanation}</p>
          </div>
        ))}
        
        {analysis.length>0 && (
        <div>
          <h2>Analysis Results (This Analysis is an AI generated financial analysis using income statements, cashflow statements and balance sheet of the company obtained from EDGAR SEC filings)</h2>
          <p style={{fontSize: '1.1rem', textAlign: 'justify'}}>{analysis}</p>
        </div>
        )}

      </div>
      )};

    </div>
  );
}

export default App;