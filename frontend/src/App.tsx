import { ChangeEvent, useState } from 'react'
import { Activity, ArrowUpRight, CheckCircle2, FileImage, ScanSearch, ShieldCheck, Upload, Zap } from 'lucide-react'

type ApiReport = {
  label: string
  synthetic_likelihood: number
  spatial_signal: number
  frequency_signal: number
  compression_response: number
  status: string
  note: string
}

const demoReport: ApiReport = {
  label: 'Likely synthetic',
  synthetic_likelihood: 0.76,
  spatial_signal: 0.84,
  frequency_signal: 0.71,
  compression_response: 0.63,
  status: 'demo',
  note: 'Demo result. Train a checkpoint and connect it to replace this estimate with model inference.',
}

function App() {
  const [fileName, setFileName] = useState('portrait_study.jpg')
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [scanned, setScanned] = useState(false)
  const [isScanning, setIsScanning] = useState(false)
  const [error, setError] = useState('')
  const [report, setReport] = useState<ApiReport>(demoReport)

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
      setReport(await response.json())
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
          <div className="panel-heading"><div><span className="section-number">02</span><h2>Forensic report</h2></div><span className="report-state">{scanned ? 'JUST NOW' : 'DEMO RESULT'}</span></div>
          <div className="verdict-row"><div><p className="eyebrow">Synthetic likelihood</p><div className="score">{Math.round(report.synthetic_likelihood * 100)}<span>%</span></div></div><div className="verdict-badge">{report.label}</div></div>
          <div className="confidence-track"><span style={{ width: `${report.synthetic_likelihood * 100}%` }} /></div>
          <p className="result-summary">{report.note} This estimate is evidence, not proof of origin.</p>
          <div className="evidence-list">{[
            { label: 'Spatial signal', value: report.spatial_signal * 100, note: 'Texture and edge patterns', color: 'coral' },
            { label: 'Frequency signal', value: report.frequency_signal * 100, note: 'FFT spectral residue', color: 'gold' },
            { label: 'Compression response', value: report.compression_response * 100, note: 'JPEG quality 70', color: 'teal' },
          ].map((item) => <div className="evidence-item" key={item.label}><div className="evidence-copy"><span>{item.label}</span><small>{item.note}</small></div><strong>{Math.round(item.value)}<sup>%</sup></strong><div className={`mini-track ${item.color}`}><span style={{ width: `${item.value}%` }} /></div></div>)}</div>
          <div className="experiment-note"><span className="note-pin" /><div><strong>Research context</strong><p>Fusion model trained on real + generators A/B. Generator C remains held out for evaluation.</p></div></div>
        </div>
      </section>

      <footer><span>TRACE / EXPLAINABLE IMAGE FORENSICS</span><span>NOT LEGAL OR ABSOLUTE PROOF OF AUTHENTICITY</span><span>◎ 2026</span></footer>
    </main>
  )
}

export default App
