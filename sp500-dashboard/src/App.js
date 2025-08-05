import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [companies, setCompanies] = useState([]);
  const [selected, setSelected] = useState('');
  const [plots, setPlots] = useState([]);
  const FLASK_BACKEND_URL = 'https://financial-data-analysis.onrender.com';
  const FLASK_BACKEND_PLOT_URL = 'https://financial-data-analysis.onrender.com/static/plots/';


  useEffect(() => {
    axios.get(`${FLASK_BACKEND_URL}/api/companies`).then(res => setCompanies(res.data));
  }, []);

  const handleChange = (e) => { 
    setSelected(e.target.value);
    axios.get(`${FLASK_BACKEND_URL}/api/analysis/${e.target.value}`).then(res => setPlots(res.data.plots));
  };

  return (
    <div>
      <h1>S&P 500 Company Analysis Dashboard</h1>
      <select onChange={handleChange} value={selected}>
        <option value="">Select a company</option>
        {companies.map(c => (
          <option key={c.tic} value={c.tic}>{c.conm} ({c.tic})</option>
        ))}
      </select>
      <div>
        {plots.map(plot => (
          <div key={plot.title}>
            <h3>{plot.title}</h3>
            <img src={`${FLASK_BACKEND_PLOT_URL}${plot.url}`} alt={plot.title} style={{maxWidth: '600px'}} />
            <p>{plot.explanation}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;