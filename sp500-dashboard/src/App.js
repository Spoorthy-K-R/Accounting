import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [companies, setCompanies] = useState([]);
  const [selected, setSelected] = useState('');
  const [plots, setPlots] = useState([]);
  const [analysis, setAnalysis] = useState([]);

  const [isLoading, setIsLoading] = useState(false); // New state for loading indicator
  const [error, setError] = useState(null); // New state for error messages


  const FLASK_BACKEND_URL = 'https://financial-data-analysis.onrender.com';
  // const FLASK_BACKEND_URL = 'http://127.0.0.1:5000';


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
    setAnalysis(null);
    setError(null);
    
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

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', padding: '20px', maxWidth: '900px', margin: '0 auto', textAlign: 'center' }}>
      <h1>S&P 500 Company Analysis Dashboard</h1>
      
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
      </div>

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

      {!isLoading && !error && plots.length === 0 && selected && (
        <div style={{ padding: '20px', color: '#777' }}>
          <p>No analysis available for {selected} or data is still processing.</p>
        </div>
      )}

      {!isLoading && !error && plots.length === 0 && !selected && (
        <div style={{ padding: '20px', color: '#777' }}>
          <p>Select a company from the dropdown to view its stock analysis dashboard.</p>
        </div>
      )}

      {!isLoading && !error && plots.length > 0 && (
        <div style={{ marginTop: '20px' }}>
        <h2>Analysis Results for {selected}</h2>

        {plots.map(plot => (
            <div key={plot.title} style={{ border: '1px solid #eee', borderRadius: '8px', padding: '15px', marginBottom: '25px', backgroundColor: '#f9f9f9' }}>
              <h3 style={{ color: '#333', marginBottom: '10px' }}>{plot.title}</h3>
              <img src={`${FLASK_BACKEND_URL}${plot.url}`} alt={plot.title} style={{maxWidth: '100%', height: 'auto', borderRadius: '4px'}} />
              <p style={{ fontSize: '0.95em', color: '#666', marginTop: '15px', lineHeight: '1.4' }}>{plot.explanation}</p>
          </div>
        ))}
        
        { analysis.length>0 && (
        <div>
          <h2>Analysis Results (This Analysis is AI generated)</h2>
          <p style={{fontSize: '1.1rem', textAlign: 'justify'}}>{analysis}</p>
        </div>
        )}

      </div>
      )};

    </div>
  );
}

export default App;