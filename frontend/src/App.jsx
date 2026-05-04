import { useMemo, useState } from 'react'

const API_BASE = 'http://127.0.0.1:8000'

const EXERCISES = ['abs', 'back', 'bicep_curl', 'push_up', 'shoulder', 'squat', 'tricep']
const FORM_QUALITY = ['good', 'bad']
const ANGLES = ['front', 'side', 'diagonal']

export default function App() {
  const [video, setVideo] = useState(null)
  const [subjectId, setSubjectId] = useState('subject_001')
  const [exercise, setExercise] = useState('squat')
  const [formQuality, setFormQuality] = useState('good')
  const [angleView, setAngleView] = useState('front')
  const [clipLimit, setClipLimit] = useState('2.0')
  const [tileSize, setTileSize] = useState('8')
  const [minVisibility, setMinVisibility] = useState('0.5')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const canSubmit = useMemo(() => !!video && !loading, [video, loading])

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    setResult(null)

    if (!video) {
      setError('Please select a video file.')
      return
    }

    const fd = new FormData()
    fd.append('video', video)
    fd.append('subject_id', subjectId)
    fd.append('exercise_folder', exercise)
    fd.append('form_quality', formQuality)
    fd.append('angle_view', angleView)
    fd.append('clip_limit', clipLimit)
    fd.append('tile_size', tileSize)
    fd.append('min_visibility', minVisibility)

    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/preprocess`, {
        method: 'POST',
        body: fd,
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || 'Preprocessing failed')
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="bg-shape bg1" />
      <div className="bg-shape bg2" />

      <main className="card">
        <header>
          <h1>FitAI Preprocessing Tool</h1>
          <p>Upload one exercise video and download the preprocessed CSV instantly.</p>
        </header>

        <form onSubmit={onSubmit} className="form">
          <label className="file-input">
            <span>Video File</span>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setVideo(e.target.files?.[0] || null)}
              required
            />
            {video ? <small>{video.name}</small> : <small>No file selected</small>}
          </label>

          <div className="grid">
            <label>
              <span>Subject ID</span>
              <input value={subjectId} onChange={(e) => setSubjectId(e.target.value)} required />
            </label>

            <label>
              <span>Exercise</span>
              <select value={exercise} onChange={(e) => setExercise(e.target.value)}>
                {EXERCISES.map((x) => (
                  <option key={x} value={x}>{x}</option>
                ))}
              </select>
            </label>

            <label>
              <span>Form Quality</span>
              <select value={formQuality} onChange={(e) => setFormQuality(e.target.value)}>
                {FORM_QUALITY.map((x) => (
                  <option key={x} value={x}>{x}</option>
                ))}
              </select>
            </label>

            <label>
              <span>Angle View</span>
              <select value={angleView} onChange={(e) => setAngleView(e.target.value)}>
                {ANGLES.map((x) => (
                  <option key={x} value={x}>{x}</option>
                ))}
              </select>
            </label>

            <label>
              <span>CLAHE Clip Limit</span>
              <input value={clipLimit} onChange={(e) => setClipLimit(e.target.value)} />
            </label>

            <label>
              <span>CLAHE Tile Size</span>
              <input value={tileSize} onChange={(e) => setTileSize(e.target.value)} />
            </label>

            <label>
              <span>Min Visibility</span>
              <input value={minVisibility} onChange={(e) => setMinVisibility(e.target.value)} />
            </label>
          </div>

          <button className="primary" disabled={!canSubmit}>
            {loading ? 'Processing...' : 'Generate CSV'}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {result && (
          <section className="result">
            <h2>Processing Complete</h2>
            <p><strong>Job ID:</strong> {result.job_id}</p>
            <p><strong>Subject:</strong> {result.report.subject_id}</p>
            <p><strong>Exercise:</strong> {result.report.exercise_folder} ({result.report.canonical_exercise})</p>
            <p><strong>Frames:</strong> {result.report.kept_frames} kept / {result.report.total_frames} total</p>
            <p><strong>Kept Ratio:</strong> {result.report.kept_ratio}</p>

            <div className="actions">
              <a className="download" href={`${API_BASE}${result.download_url}`} target="_blank" rel="noreferrer">
                Download CSV
              </a>
              <a className="download secondary" href={`${API_BASE}${result.report_url}`} target="_blank" rel="noreferrer">
                Download Report JSON
              </a>
            </div>
          </section>
        )}
      </main>
    </div>
  )
}
