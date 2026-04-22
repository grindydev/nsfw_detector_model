import { useState, useRef, useCallback, useEffect } from 'react'
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
  const [mode, setMode] = useState(null)              // 'image' | 'video'
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)          // image API response
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef(null)

  // Video scanning state
  const [videoStatus, setVideoStatus] = useState('idle')  // idle | extracting | scanning | done
  const [videoProgress, setVideoProgress] = useState(0)
  const [videoTotalFrames, setVideoTotalFrames] = useState(0)
  const [videoDuration, setVideoDuration] = useState(0)
  const [flaggedFrames, setFlaggedFrames] = useState([])
  const [currentFrame, setCurrentFrame] = useState(null)
  const [isNsfw, setIsNsfw] = useState(false)
  const [revealedFrames, setRevealedFrames] = useState(new Set())
  const [filterCategory, setFilterCategory] = useState('all')

  // Model info from backend
  const [modelInfo, setModelInfo] = useState(null)

  useEffect(() => {
    fetch('/model-info')
      .then(res => res.json())
      .then(data => { if (data.loaded) setModelInfo(data) })
      .catch(() => {})
  }, [])

  const reset = useCallback(() => {
    setMode(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setResult(null)
    setLoading(false)
    setError(null)
    setVideoStatus('idle')
    setVideoProgress(0)
    setVideoTotalFrames(0)
    setVideoDuration(0)
    setFlaggedFrames([])
    setCurrentFrame(null)
    setIsNsfw(false)
    setRevealedFrames(new Set())
    setFilterCategory('all')
  }, [previewUrl])

  // ─── Image prediction ───────────────────────────
  const handleImage = async (file) => {
    setMode('image')
    setPreviewUrl(URL.createObjectURL(file))
    setLoading(true)
    setError(null)

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

  // ─── Video scanning ─────────────────────────────
  const handleVideo = async (file) => {
    setMode('video')
    setPreviewUrl(URL.createObjectURL(file))
    setLoading(true)
    setError(null)
    setVideoStatus('extracting')
    setVideoProgress(0)
    setFlaggedFrames([])
    setCurrentFrame(null)
    setIsNsfw(false)
    setRevealedFrames(new Set())

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/predict-video', { method: 'POST', body: formData })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Video scan failed')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))

            switch (data.type) {
              case 'extracting':
                setVideoStatus('extracting')
                break

              case 'info':
                setVideoTotalFrames(data.total_frames)
                setVideoDuration(data.duration)
                setVideoStatus('scanning')
                break

              case 'frame': {
                const progress = Math.round((data.frame / data.total) * 100)
                setVideoProgress(progress)
                setCurrentFrame(data)

                if (data.nsfw) {
                  setIsNsfw(true)
                  setFlaggedFrames(prev => [...prev, data])
                }
                break
              }

              case 'done':
                setVideoProgress(100)
                setVideoStatus('done')
                setLoading(false)
                break

              case 'error':
                setError(data.message || 'Unknown error')
                break
            }
          } catch {
            // skip malformed JSON
          }
        }
      }
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  // ─── File routing ───────────────────────────────
  const handleFile = (file) => {
    if (!file) return
    if (file.type.startsWith('image/')) {
      handleImage(file)
    } else if (file.type.startsWith('video/')) {
      handleVideo(file)
    } else {
      setError('Please select an image or video file')
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const toggleReveal = (idx) => {
    setRevealedFrames(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  // Filter flagged frames by selected category
  const filteredFrames = filterCategory === 'all'
    ? flaggedFrames
    : flaggedFrames.filter(f => (f.nsfw_classes || {})[filterCategory] > 0)

  // Count frames per NSFW category
  const nsfwCounts = { all: flaggedFrames.length }
  for (const f of flaggedFrames) {
    for (const cls of Object.keys(f.nsfw_classes || {})) {
      nsfwCounts[cls] = (nsfwCounts[cls] || 0) + 1
    }
  }

  // ─── Render helpers ─────────────────────────────
  const renderImageResults = () => (
    <div className="results">
      <div className="prediction">
        <span className="prediction-label">Prediction:</span>
        <span className="prediction-value" style={{ color: CLASS_COLORS[result.prediction] }}>
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
  )

  const renderVideoResults = () => (
    <div className="video-results">
      {/* Video player */}
      {previewUrl && (
        <div className="video-player">
          <video src={previewUrl} controls muted />
        </div>
      )}

      {/* NSFW warning banner */}
      {isNsfw && (
        <div className="nsfw-warning">
          <span className="warning-icon">⚠️</span>
          <div>
            <strong>NSFW Content Detected</strong>
            <p>{flaggedFrames.length} frame{flaggedFrames.length !== 1 ? 's' : ''} flagged</p>
          </div>
        </div>
      )}

      {/* Extracting status */}
      {videoStatus === 'extracting' && (
        <div className="progress-section">
          <div className="progress-header">
            <span>⏳ Extracting frames from video...</span>
          </div>
          <div className="progress-bar-bg">
            <div className="progress-bar-fill extracting" />
          </div>
        </div>
      )}

      {/* Scanning progress */}
      {videoStatus === 'scanning' && (
        <div className="progress-section">
          <div className="progress-header">
            <span>🔍 Scanning frames... {currentFrame && (
              <span className="current-class" style={{ color: CLASS_COLORS[currentFrame.prediction] }}>
                {CLASS_LABELS[currentFrame.prediction]}
              </span>
            )}</span>
            <span>{videoProgress}%</span>
          </div>
          <div className="progress-bar-bg">
            <div
              className="progress-bar-fill"
              style={{ width: `${videoProgress}%` }}
            />
          </div>
          <p className="progress-detail">
            Frame {currentFrame?.frame || 0} of {videoTotalFrames}
            {videoDuration > 0 && ` • ${videoDuration}s video`}
            {currentFrame && ` • @${currentFrame.timestamp}s`}
          </p>
        </div>
      )}

      {/* Done - safe */}
      {videoStatus === 'done' && !isNsfw && (
        <div className="safe-banner">
          <span>✅</span>
          <strong>No NSFW content detected</strong>
          <p>{videoTotalFrames} frames scanned</p>
        </div>
      )}

      {/* Flagged frames with images */}
      {flaggedFrames.length > 0 && (
        <div className="flagged-section">
          <div className="flagged-header">
            <h3>🚩 Flagged Frames ({filteredFrames.length})</h3>
            <div className="filter-tabs">
              <button
                className={`filter-tab ${filterCategory === 'all' ? 'active' : ''}`}
                onClick={() => setFilterCategory('all')}
              >
                All ({nsfwCounts.all})
              </button>
              {['hentai', 'porn', 'sexy'].map(cls => (
                nsfwCounts[cls] ? (
                  <button
                    key={cls}
                    className={`filter-tab ${filterCategory === cls ? 'active' : ''}`}
                    onClick={() => setFilterCategory(cls)}
                    style={filterCategory === cls ? { borderColor: CLASS_COLORS[cls], color: CLASS_COLORS[cls] } : {}}
                  >
                    {CLASS_LABELS[cls]} ({nsfwCounts[cls]})
                  </button>
                ) : null
              ))}
            </div>
          </div>
          <div className="flagged-grid">
            {filteredFrames.map((f, idx) => {
              const originalIdx = flaggedFrames.indexOf(f)
              return (
                <div
                  key={originalIdx}
                  className={`flagged-card ${revealedFrames.has(originalIdx) ? 'revealed' : ''}`}
                  onClick={() => toggleReveal(originalIdx)}
                >
                  <div className="flagged-thumb">
                    <img
                      src={`data:image/jpeg;base64,${f.image}`}
                      alt={`Frame at ${f.timestamp}s`}
                    />
                    {!revealedFrames.has(originalIdx) && <div className="blur-overlay" />}
                  </div>
                  <div className="flagged-info">
                    <span className="flagged-time">@{f.timestamp}s</span>
                    <div className="flagged-badges">
                      {Object.entries(f.nsfw_classes || {}).map(([cls, conf]) => (
                        <span
                          key={cls}
                          className="flagged-badge"
                          style={{ backgroundColor: CLASS_COLORS[cls] + '33', color: CLASS_COLORS[cls] }}
                        >
                          {CLASS_LABELS[cls]} {(conf * 100).toFixed(0)}%
                        </span>
                      ))}
                    </div>
                    <span className="reveal-hint">
                      {revealedFrames.has(originalIdx) ? '👁 Hide' : '🔒 Click to reveal'}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )

  return (
    <div className="app">
      <div className="container">
        <h1>🔍 NSFW Detector</h1>
        <p className="subtitle">Upload an image or video to classify its content</p>
        {modelInfo && (
          <div className="model-info">
            <span className="model-badge model-name">{modelInfo.model_type}</span>
            <span className="model-badge">📦 {modelInfo.model_size_kb.toFixed(0)} KB</span>
            <span className="model-badge">🖼 {modelInfo.input_size}×{modelInfo.input_size}</span>
          </div>
        )}

        {!previewUrl ? (
          <div
            className={`dropzone ${dragging ? 'dragging' : ''}`}
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-content">
              <span className="dropzone-icon">📁</span>
              <p>Drop an image or video here or <strong>click to browse</strong></p>
              <p className="dropzone-hint">Supports JPG, PNG, WebP, MP4, WebM, MOV</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*"
              onChange={(e) => handleFile(e.target.files[0])}
              hidden
            />
          </div>
        ) : (
          <div className="results-area">
            {/* Image preview */}
            {mode === 'image' && previewUrl && (
              <div className="image-preview">
                <img src={previewUrl} alt="Uploaded" />
              </div>
            )}

            {/* Loading spinner for image */}
            {mode === 'image' && loading && (
              <div className="loading">
                <div className="spinner"></div>
                <p>Analyzing image...</p>
              </div>
            )}

            {/* Image results */}
            {mode === 'image' && result && renderImageResults()}

            {/* Video results */}
            {mode === 'video' && renderVideoResults()}

            {/* Error */}
            {error && (
              <div className="error">
                <p>❌ {error}</p>
              </div>
            )}

            <button className="reset-btn" onClick={reset}>
              {mode === 'video' ? 'Scan Another' : 'Upload Another Image'}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
