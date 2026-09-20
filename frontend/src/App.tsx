import { ChangeEvent, useState } from 'react'
import { Activity, ArrowUpRight, CheckCircle2, FileImage, ScanSearch, ShieldCheck, Upload, Zap } from 'lucide-react'

type SignalDirection = 'synthetic' | 'authentic' | 'ambiguous'

type RobustnessPoint = {
  transform: string
  score: number
}

type ForensicReport = {
  syntheticEvidenceScore: number | null
  prediction: string
  spatialScore: number | null
  frequencyScore: number | null
  fusionScore: number | null
  evidenceDirection: { spatial: SignalDirection; frequency: SignalDirection }
  calibration: { status: string; ece: number | null }
  robustness: RobustnessPoint[]
  evaluationContext: { trainingGenerators: string[]; seenGenerators: string[]; unseenGenerators: string[] }
  spatialAttribution: string | null
  frequencyVisualization: string | null
  status: string
  note: string
}

type LegacyApiReport = {
  label?: string
  synthetic_likelihood?: number
  spatial_signal?: number
  frequency_signal?: number
  compression_response?: number
  status?: string
  note?: string
  synthetic_evidence_score?: number | null
  prediction?: string
  spatial_score?: number
  frequency_score?: number
  fusion_score?: number
  evidence_direction?: { spatial?: SignalDirection; frequency?: SignalDirection }
  calibration?: { status?: string; ece?: number | null }
  robustness?: RobustnessPoint[]
  evaluation_context?: { training_generators?: string[]; seen_generators?: string[]; unseen_generators?: string[] }
  spatial_attribution?: string | null
  frequency_visualization?: string | null
}

const demoReport: ForensicReport = {
  syntheticEvidenceScore: 76,
  prediction: 'Likely synthetic',
  spatialScore: 84,
  frequencyScore: 71,
  fusionScore: 76,
  evidenceDirection: { spatial: 'synthetic', frequency: 'synthetic' },
  calibration: { status: 'Not yet validated', ece: null },
  robustness: [],
  evaluationContext: { trainingGenerators: [], seenGenerators: [], unseenGenerators: [] },
  spatialAttribution: null,
  frequencyVisualization: null,
  status: 'demo',
  note: 'DEMO / SMOKE-TEST OUTPUT — NOT VALIDATED. Train and evaluate a research checkpoint before interpreting this report.',
}

function directionFromLabel(label?: string): SignalDirection {
  const value = label?.toLowerCase() ?? ''
  if (value.includes('authentic')) return 'authentic'
  if (value.includes('synthetic')) return 'synthetic'
  return 'ambiguous'
}

function normalizeReport(raw: LegacyApiReport): ForensicReport {
  const evidenceScore = 'synthetic_evidence_score' in raw ? raw.synthetic_evidence_score ?? null : (raw.synthetic_likelihood != null ? raw.synthetic_likelihood * 100 : null)
  return {
    syntheticEvidenceScore: evidenceScore,
    prediction: raw.prediction ?? raw.label ?? 'Ambiguous',
    spatialScore: raw.spatial_score ?? (raw.spatial_signal != null ? raw.spatial_signal * 100 : null),
    frequencyScore: raw.frequency_score ?? (raw.frequency_signal != null ? raw.frequency_signal * 100 : null),
    fusionScore: raw.fusion_score ?? evidenceScore,
    evidenceDirection: {
      spatial: raw.evidence_direction?.spatial ?? directionFromLabel(raw.label),
      frequency: raw.evidence_direction?.frequency ?? directionFromLabel(raw.label),
    },
    calibration: { status: raw.calibration?.status ?? 'Not yet validated', ece: raw.calibration?.ece ?? null },
    robustness: raw.robustness ?? [],
    evaluationContext: {
      trainingGenerators: raw.evaluation_context?.training_generators ?? [],
      seenGenerators: raw.evaluation_context?.seen_generators ?? [],
      unseenGenerators: raw.evaluation_context?.unseen_generators ?? [],
    },
    spatialAttribution: raw.spatial_attribution ?? null,
    frequencyVisualization: raw.frequency_visualization ?? null,
    status: raw.status ?? 'inference',
    note: raw.note ?? 'No structured explanation was returned by the analysis service.',
  }
}

function formatScore(score: number | null, suffix = '/ 100') {
  return score == null ? '—' : `${Math.round(score)} ${suffix}`
}

function formatDirection(direction: SignalDirection) {
  return direction === 'synthetic' ? '→ Synthetic' : direction === 'authentic' ? '→ Authentic' : '→ Ambiguous'
}

function App() {
  const [fileName, setFileName] = useState('portrait_study.jpg')
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [scanned, setScanned] = useState(false)
  const [isScanning, setIsScanning] = useState(false)
  const [error, setError] = useState('')
  const [report, setReport] = useState<ForensicReport>(demoReport)

  function acceptFile(file?: File) {
    if (!file) return
    setFileName(file.name)
    setFile(file)
    setScanned(false)
    setError('')
  }

  async function runScan() {
    if (!file) {
      setError('Choose an image before running a scan.')
      return
    }
    setIsScanning(true)
    setError('')
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await fetch('http://localhost:8000/scan', { method: 'POST', body: formData })
      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: 'The analysis service is unavailable.' }))
        throw new Error(body.detail)
      }
      setReport(normalizeReport(await response.json()))
      setScanned(true)
    } catch (scanError) {
      setError(scanError instanceof Error ? scanError.message : 'The analysis service is unavailable.')
    } finally {
      setIsScanning(false)
    }
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0])
  }

  return (
    <main className="app-shell">
      <nav className="topbar">
        <div className="brand"><span className="brand-mark"><ScanSearch size={19} /></span><span>TRACE</span></div>
        <div className="nav-status"><span className="status-dot" /> analysis engine online <span className="version">v0.1 / research build</span></div>
        <button className="icon-button" aria-label="Activity log"><Activity size={18} /></button>
      </nav>

      <section className="intro">
        <div>
          <p className="eyebrow">Image forensics / workspace 01</p>
          <h1>Look closer.</h1>
          <p className="lede">Evidence-led analysis for images that deserve a second look.</p>
        </div>
        <div className="intro-aside"><span className="line" /> <span>SPATIAL + FREQUENCY<br />UNSEEN-GENERATOR READY</span></div>
      </section>

      <section className="workbench">
        <div className="upload-panel">
          <div className="panel-heading"><div><span className="section-number">01</span><h2>Submit an image</h2></div><span className="quiet-label">PNG / JPG / WEBP</span></div>
          <label className={`drop-zone ${isDragging ? 'is-dragging' : ''}`} onDragOver={(event: { preventDefault: () => void }) => { event.preventDefault(); setIsDragging(true) }} onDragLeave={() => setIsDragging(false)} onDrop={(event: { preventDefault: () => void; dataTransfer: DataTransfer }) => { event.preventDefault(); setIsDragging(false); acceptFile(event.dataTransfer.files[0]) }}>
            <input type="file" accept="image/png,image/jpeg,image/webp" onChange={handleChange} />
            <span className="upload-icon"><Upload size={22} /></span>
            <strong>Drop an image here</strong>
            <span>or browse from your device</span>
          </label>
          <div className="selected-file"><FileImage size={17} /><span>{fileName}</span><CheckCircle2 size={16} /></div>
          <button className="scan-button" onClick={runScan} disabled={isScanning}><Zap size={17} /> {isScanning ? 'Analyzing image...' : scanned ? 'Analysis complete' : 'Run forensic scan'} <ArrowUpRight size={17} /></button>
          {error && <p className="scan-error">{error}</p>}
          <p className="privacy-note"><ShieldCheck size={14} /> Files are analyzed temporarily. Only a hash and report are retained.</p>
        </div>

        <div className="result-panel">
          <div className="panel-heading"><div><span className="section-number">02</span><h2>Forensic report</h2></div><span className="report-state">{scanned ? 'JUST NOW' : 'RESEARCH BUILD'}</span></div>
          <div className="verdict-row"><div><p className="eyebrow">Synthetic evidence score</p><div className="score">{report.syntheticEvidenceScore == null ? '—' : Math.round(report.syntheticEvidenceScore)}<span>/ 100</span></div></div><div className="verdict-badge">{report.prediction}</div></div>
          <div className="confidence-track"><span style={{ width: `${report.syntheticEvidenceScore ?? 0}%` }} /></div>
          <p className="result-summary">{report.note}</p>

          <div className="report-section">
            <div className="report-section-heading"><span className="eyebrow">Evidence breakdown</span><span className="quiet-label">BRANCH OUTPUTS</span></div>
            <div className="evidence-list">
              {[
                { label: 'Spatial signal', value: report.spatialScore, note: 'Texture and edge patterns', color: 'coral' },
                { label: 'Frequency signal', value: report.frequencyScore, note: 'FFT / DCT spectral statistics', color: 'gold' },
                { label: 'Fusion signal', value: report.fusionScore, note: 'Combined spatial + frequency evidence', color: 'teal' },
              ].map((item) => <div className="evidence-item" key={item.label}><div className="evidence-copy"><span>{item.label}</span><small>{item.note}</small></div><strong>{formatScore(item.value)}</strong><div className={`mini-track ${item.color}`}><span style={{ width: `${item.value ?? 0}%` }} /></div></div>)}
            </div>
          </div>

          <div className="report-section direction-section">
            <div className="report-section-heading"><span className="eyebrow">Evidence direction</span><span className="quiet-label">BRANCH READOUT</span></div>
            <div className="direction-grid"><span>Spatial <strong>{formatDirection(report.evidenceDirection.spatial)}</strong></span><span>Frequency <strong>{formatDirection(report.evidenceDirection.frequency)}</strong></span></div>
          </div>

          <details className="report-details" open={report.robustness.length > 0}>
            <summary><span><span className="eyebrow">Robustness</span><small>Perturbation stability, not synthetic evidence</small></span><span className="details-chevron">v</span></summary>
            {report.robustness.length > 0 ? <div className="robustness-list">{report.robustness.map((point) => <div className="robustness-row" key={point.transform}><span>{point.transform.split('_').join(' ').toUpperCase()}</span><strong>{formatScore(point.score, '')}</strong></div>)}<p className="stability-label">STABILITY: {report.robustness.length > 1 ? 'REPORTED' : 'LIMITED DATA'}</p></div> : <p className="empty-report">No perturbation series returned by the backend yet.</p>}
          </details>

          <div className="report-section status-grid">
            <div><span className="eyebrow">Calibration</span><strong>{report.calibration.status}</strong>{report.calibration.ece != null && <small>ECE: {report.calibration.ece.toFixed(2)}</small>}</div>
            <div><span className="eyebrow">Model context</span><strong>{report.status === 'inference' ? 'Validated inference' : 'Unvalidated baseline'}</strong><small>{report.status === 'inference' ? 'Context supplied by the analysis service.' : 'Research evaluation has not yet been completed.'}</small></div>
          </div>

          <div className="visual-evidence-grid">
            <div className="visual-placeholder"><span className="eyebrow">Spatial attribution</span><Activity size={19} /><small>{report.spatialAttribution ?? 'Attribution map not returned yet.'}</small><em>Model attribution — not a ground-truth manipulation mask.</em></div>
            <div className="visual-placeholder"><span className="eyebrow">Frequency evidence</span><Activity size={19} /><small>{report.frequencyVisualization ?? 'Frequency statistics not returned yet.'}</small><em>Visualization will reflect measured bands, not proof.</em></div>
          </div>

          <div className="experiment-note model-context"><span className="note-pin" /><div><strong>Evaluation context</strong><p>Training: {report.evaluationContext.trainingGenerators.length ? report.evaluationContext.trainingGenerators.join(' + ') : 'Not reported'}</p><p>Seen evaluation: {report.evaluationContext.seenGenerators.length ? report.evaluationContext.seenGenerators.join(' + ') : 'Not reported'}</p><p className="unseen-label">Unseen generator: {report.evaluationContext.unseenGenerators.length ? report.evaluationContext.unseenGenerators.join(', ') : 'Not reported'} / held out during development</p></div></div>
        </div>
      </section>

      <footer><span>TRACE / EXPLAINABLE IMAGE FORENSICS</span><span>TRACE ESTIMATES LEARNED FORENSIC EVIDENCE ASSOCIATED WITH SYNTHETIC IMAGERY. IT DOES NOT ESTABLISH IMAGE PROVENANCE, AUTHORSHIP, OR AUTHENTICITY.</span><span>NOT LEGAL OR ABSOLUTE PROOF OF AUTHENTICITY</span><span>◎ 2026</span></footer>
    </main>
  )
}

export default App
