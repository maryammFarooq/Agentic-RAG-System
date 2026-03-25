// Implement: query input, pipeline selector (RAG Fusion, HyDE, CRAG, Graph RAG), Run button, display retrieved chunks and answer.
// Main React imports
import React, { useState } from 'react';
import './modern.css';


// Main App component for the RAG UI
function App() {
  // User's input question
  const [userQuestion, setUserQuestion] = useState('');
  // Selected pipeline strategy
  const [selectedPipeline, setSelectedPipeline] = useState('rag_fusion');
  // Loading state for async requests
  const [isLoading, setIsLoading] = useState(false);
  // Stores the result from the backend
  const [pipelineResult, setPipelineResult] = useState(null);
  // Error message state
  const [errorMsg, setErrorMsg] = useState('');

  /**
   * Handles form submission: sends the query and pipeline to the backend API.
   * @param {Event} e - Form submit event
   */
  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!userQuestion.trim()) return;

    setIsLoading(true);
    setErrorMsg('');
    setPipelineResult(null);

    try {
      // Send POST request to backend API
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userQuestion, pipeline: selectedPipeline }),
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      // Parse JSON response
      const data = await response.json();
      setPipelineResult(data);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header and logo/title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '18px' }}>
        {/* Logo area (optional: add an <img> if you have a logo) */}
        <span style={{ fontSize: '2.2rem', color: '#1976d2', fontWeight: 700, letterSpacing: '2px' }}>RAG<span style={{ color: '#42a5f5' }}>Lab</span></span>
      </div>
      <h1>RAG Pipeline Evaluator</h1>
      <p>Test and compare retrieval-augmented generation strategies on financial and general knowledge queries.</p>

      {/* Query form */}
      <form onSubmit={handleQuerySubmit}>
        {/* Pipeline selector dropdown */}
        <div>
          <label htmlFor="pipeline-select"><strong>Select Strategy:</strong></label>
          <select 
            id="pipeline-select"
            value={selectedPipeline} 
            onChange={(e) => setSelectedPipeline(e.target.value)}
          >
            <option value="rag_fusion">RAG Fusion</option>
            <option value="hyde">HyDE</option>
            <option value="crag">CRAG (Corrective RAG)</option>
            <option value="graph_rag">Graph RAG</option>
          </select>
        </div>

        {/* User question input */}
        <div>
          <textarea
            value={userQuestion}
            onChange={(e) => setUserQuestion(e.target.value)}
            placeholder="e.g., Which athlete has won more Grand Slams, Federer or Nadal?"
            rows={3}
            style={{ width: '100%' }}
          />
        </div>

        {/* Submit button */}
        <button 
          type="submit" 
          disabled={isLoading}
        >
          {isLoading ? 'Running Pipeline...' : 'Generate Answer'}
        </button>
      </form>

      {/* Error message display */}
      {errorMsg && (
        <div className="error">
          <strong>Error:</strong> {errorMsg}
        </div>
      )}

      {/* Results display */}
      {pipelineResult && (
        <div className="results-container">
          {/* Answer section */}
          <div>
            <h2>Generated Answer</h2>
            <p style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>{pipelineResult.answer}</p>
          </div>
          {/* Score/metric section */}
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Pipeline Metric</span>
              <span style={{ backgroundColor: '#28a745', color: 'white', padding: '4px 8px', borderRadius: '4px', fontSize: '0.9em' }}>
                Score / Confidence: {pipelineResult.score.toFixed(4)}
              </span>
            </h3>
            <p style={{ fontSize: '0.9em', color: '#555' }}>
              <em>(Note: RAG Fusion, HyDE, and Graph RAG show retrieval similarity scores. CRAG shows the LLM Judge confidence score.)</em>
            </p>
          </div>
          {/* Retrieved context section */}
          <div>
            <h3>Retrieved Context (Global Index)</h3>
            <pre>
              {pipelineResult.context || "No context retrieved or context skipped due to low confidence."}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
