import { useState } from 'react'

function App() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleSearch = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      // Point this to your FastAPI backend
      const res = await fetch('http://localhost:8001/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query })
      })
      const data = await res.json()
      setResult(data)
    } catch (err) {
      console.error(err)
      setResult({ error: "Failed to connect to backend." })
    }
    setLoading(false)
  }

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
      <h1>🏛️ RegWatcher AI</h1>
      <p style={{ color: '#555' }}>Ask natural language questions about US Federal Regulations.</p>
      
      <form onSubmit={handleSearch} style={{ display: 'flex', gap: '1rem', marginTop: '2rem' }}>
        <input 
          type="text" 
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. Find all documents by the EPA in 2024..."
          style={{ flex: 1, padding: '0.75rem', borderRadius: '8px', border: '1px solid #ccc', fontSize: '1rem' }}
        />
        <button 
          type="submit" 
          disabled={loading}
          style={{ padding: '0.75rem 1.5rem', borderRadius: '8px', border: 'none', background: '#0066cc', color: 'white', cursor: 'pointer', fontSize: '1rem' }}
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {result && (
        <div style={{ marginTop: '2rem', padding: '1.5rem', background: '#f5f5f5', borderRadius: '8px' }}>
          {result.error ? (
            <p style={{ color: 'red' }}>{result.error}</p>
          ) : (
            <div>
              <h3>Response</h3>
              <p>{result.response || "Check console for raw response data."}</p>
              <details style={{ marginTop: '1rem' }}>
                <summary>View Raw Data</summary>
                <pre style={{ fontSize: '0.8rem', overflowX: 'auto' }}>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
