import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [companies, setCompanies] = useState([]);
  const [selected, setSelected] = useState('');
  const [plots, setPlots] = useState([]);
  const [analysis, setAnalysis] = useState([]);
  const FLASK_BACKEND_URL = 'https://financial-data-analysis.onrender.com';
  // const FLASK_BACKEND_URL = 'http://127.0.0.1:5000';


  useEffect(() => {
    axios.get(`${FLASK_BACKEND_URL}/api/companies`).then(res => setCompanies(res.data));
  }, []);

  const handleChange = (e) => { 
    setSelected(e.target.value);
    axios.get(`${FLASK_BACKEND_URL}/api/analysis/${e.target.value}`).then(res => {
      setPlots(res.data.plots)
      setAnalysis(res.data.text)
      console.log('here')
  });
  };

  return (
    <div>
      <h1>S&P 500 Company Analysis Dashboard</h1>
      <div>
        <select onChange={handleChange} value={selected}>
          <option value="">Select a company</option>
          {companies.map(c => (
            <option key={c.tic} value={c.tic}>{c.conm} ({c.tic})</option>
          ))}
        </select>
        {/* <button onChange={handleAnalysis}>Display Analysis</button> */}
      </div>
      <div>
        {plots.map(plot => (
          <div key={plot.title}>
            <h3>{plot.title}</h3>
            <img src={`${FLASK_BACKEND_URL}${plot.url}`} alt={plot.title} style={{maxWidth: '600px'}} />
            <p>{plot.explanation}</p>
          </div>
        ))}
        { analysis.length>0 && (
        <div>
          <h2>Analysis Results</h2>
          <p>{analysis}</p>
        </div>
      ) }
      </div>
    </div>
  );
}

export default App;