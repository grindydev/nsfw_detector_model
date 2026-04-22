import { useState, useRef } from 'react'
import './App.css'

const CLASS_NAMES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

const CLASS_COLORS = {
  drawings: '#3b82f6',
  hentai:   '#8b5cf6',
  neutral:  '#6b7280',
  porn:     '#ef4444',
  sexy:     '#f97316',
}

const CLASS_LABELS = {
  drawings: '🎨 Drawings',
  hentai:   '👾 Hentai',
  neutral:  '😶 Neutral',
  porn:     '🔴 Porn',
  sexy:     '🟠 Sexy',
}

function App() {
  const [image, setImage] = useState(null)           // preview URL
  const [result, setResult] = useState(null)          // API response
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef(null)

  const handleFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please select an image file')
      return
    }

    setImage(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setLoading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/predict', { method: 'POST', body: formData })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Prediction failed')
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragging(true)
  }

  const handleDragLeave = () => {
    setDragging(false)
  }

  const reset = () => {
    setImage(null)
    setResult(null)
    setError(null)
    setLoading(false)
  }

  return (
    <div className="app">
      <div className="container">
        <h1>🔍 NSFW Detector</h1>
        <p className="subtitle">Upload an image to classify its content</p>

        {!image ? (
          <div
            className={`dropzone ${dragging ? 'dragging' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-content">
              <span className="dropzone-icon">📁</span>
              <p>Drop an image here or <strong>click to browse</strong></p>
              <p className="dropzone-hint">Supports JPG, PNG, WebP</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={(e) => handleFile(e.target.files[0])}
              hidden
            />
          </div>
        ) : (
          <div className="results-area">
            <div className="image-preview">
              <img src={image} alt="Uploaded" />
            </div>

            {loading && (
              <div className="loading">
                <div className="spinner"></div>
                <p>Analyzing image...</p>
              </div>
            )}

            {error && (
              <div className="error">
                <p>❌ {error}</p>
              </div>
            )}

            {result && (
              <div className="results">
                <div className="prediction">
                  <span className="prediction-label">Prediction:</span>
                  <span
                    className="prediction-value"
                    style={{ color: CLASS_COLORS[result.prediction] }}
                  >
                    {CLASS_LABELS[result.prediction]}
                  </span>
                  <span className="prediction-confidence">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="probabilities">
                  <h3>All Classes</h3>
                  {CLASS_NAMES.map((cls) => (
                    <div key={cls} className="prob-row">
                      <span className="prob-label">{CLASS_LABELS[cls]}</span>
                      <div className="prob-bar-bg">
                        <div
                          className="prob-bar"
                          style={{
                            width: `${result.probabilities[cls] * 100}%`,
                            backgroundColor: CLASS_COLORS[cls],
                          }}
                        />
                      </div>
                      <span className="prob-value">
                        {(result.probabilities[cls] * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <button className="reset-btn" onClick={reset}>
              Upload Another Image
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
