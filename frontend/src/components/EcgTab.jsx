import { useEffect, useMemo, useState } from 'react'
import { getEcgSamples, getEcgStatus, postEcg } from '../api.js'
import EcgChart from './EcgChart.jsx'

const COPY = {
  positive: { pill: 'Abnormal ECG likely', title: 'The model reads this ECG as abnormal',
    body: 'The model is confident enough to answer. “Abnormal” covers heart-attack patterns, ST/T changes, conduction disturbances and hypertrophy patterns; the model does not say which one.' },
  negative: { pill: 'Normal ECG likely', title: 'The model reads this ECG as normal',
    body: 'The model is confident enough to answer. A normal-looking ECG does not rule out heart disease.' },
  defer: { pill: 'Deferred to a clinician', title: 'The model is not sure enough to answer',
    body: 'Both readings remain plausible for this ECG, so the model abstains. A clinician should review the tracing.' },
}
const pct = (x) => `${Math.round(x * 100)}%`

// Accepts 1000+ rows x 12 columns, or 12 rows x 1000+ columns (comma, semicolon, tab or space separated).
export function parseEcgText(text) {
  const rows = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean)
    .map((l) => l.split(/[,;\t ]+/).map(Number))
    .filter((r) => r.length >= 12 && r.every(Number.isFinite))
  if (rows.length >= 1000 && rows[0].length === 12) return rows.slice(0, 1000)
  if (rows.length === 12 && rows[0].length >= 1000) return Array.from({ length: 1000 }, (_, t) => rows.map((r) => r[t]))
  throw new Error('Could not read the file. It needs 12 lead columns (I, II, III, aVR, aVL, aVF, V1–V6) and at least 1000 rows (10 s at 100 Hz), in millivolts.')
}

export default function EcgTab() {
  const [status, setStatus] = useState(null)
  const [samples, setSamples] = useState([])
  const [mode, setMode] = useState('sample')
  const [sel, setSel] = useState('')
  const [up, setUp] = useState({ signal: null, name: '', age: 60, sex: 1 })
  const [alpha, setAlpha] = useState(0.05)
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  useEffect(() => {
    getEcgStatus().then((s) => {
      setStatus(s)
      if (s.ready) getEcgSamples().then((r) => { setSamples(r.samples); setSel(String(r.samples[0]?.ecg_id ?? '')) }).catch((e) => setErr(e.message))
    }).catch((e) => setErr(e.message))
  }, [])

  const levels = useMemo(() => (status?.alphas ?? []), [status])

  const onFile = async (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setErr('')
    try {
      const signal = parseEcgText(await f.text())
      setUp((u) => ({ ...u, signal, name: f.name }))
    } catch (x) { setUp((u) => ({ ...u, signal: null, name: '' })); setErr(x.message) }
  }

  const run = async (e) => {
    e.preventDefault()
    setBusy(true); setErr('')
    try {
      const body = mode === 'sample' ? { ecg_id: Number(sel), alpha } : { signal: up.signal, age: Number(up.age), sex: Number(up.sex), alpha }
      setRes(await postEcg(body))
    } catch (x) { setErr(x.message) } finally { setBusy(false) }
  }

  if (err && !status) return <div className="err" role="alert">{err}</div>
  if (!status) return <p className="note">Connecting to the API…</p>
  if (!status.ready) {
    return (
      <div className="empty">
        <h2>The ECG model is not available yet</h2>
        <p>{status.detail}</p>
        <p>To enable it: <code>python -m leakfree_heart download-ptbxl</code>, then <code>python -m leakfree_heart run-ecg</code>, then restart the app.</p>
      </div>
    )
  }
  const canRun = mode === 'sample' ? !!sel : !!up.signal && up.age >= 18 && up.age <= 100

  return (
    <div className="layout">
      <form className="intake" onSubmit={run} aria-label="Choose an ECG">
        <h2>Choose an ECG</h2>
        <p className="hint">Example ECGs come from patients the model never saw during training.</p>
        <div className="seg" role="group" aria-label="ECG source">
          <button type="button" aria-pressed={mode === 'sample'} onClick={() => setMode('sample')}>Example ECG</button>
          <button type="button" aria-pressed={mode === 'upload'} onClick={() => setMode('upload')}>My own file</button>
        </div>
        {mode === 'sample' ? (
          <div className="field">
            <label htmlFor="ecg-sel">ECG</label>
            <select id="ecg-sel" value={sel} onChange={(e) => setSel(e.target.value)}>
              {samples.map((s, i) => <option key={s.ecg_id} value={s.ecg_id}>{`Example ${i + 1}: ${Math.round(s.age)} years, ${s.sex ? 'female' : 'male'}`}</option>)}
            </select>
          </div>
        ) : (
          <>
            <div className="field">
              <label htmlFor="ecg-file">ECG file (.csv or .txt)</label>
              <input id="ecg-file" type="file" accept=".csv,.txt,.tsv" onChange={onFile} style={{ gridColumn: '1 / -1' }} />
              <span className="unit" style={{ gridColumn: '1 / -1' }}>{up.signal ? `${up.name}: read 1000 samples × 12 leads` : '12 leads (I, II, III, aVR, aVL, aVF, V1–V6), 100 Hz, 10 s, millivolts.'}</span>
            </div>
            <div className="field"><label htmlFor="ecg-age">Age</label>
              <input id="ecg-age" type="number" min="18" max="100" value={up.age} onChange={(e) => setUp({ ...up, age: e.target.value })} /><span className="unit">years</span></div>
            <div className="field"><label htmlFor="ecg-sex">Sex</label>
              <select id="ecg-sex" value={up.sex} onChange={(e) => setUp({ ...up, sex: e.target.value })}>
                <option value={1}>Female</option><option value={0}>Male</option></select></div>
          </>
        )}
        <div className="group">
          <h3>How sure must the model be?</h3>
          <div className="field">
            <label htmlFor="ecg-alpha">Confidence level</label>
            <select id="ecg-alpha" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))}>
              {levels.map((l) => (
                <option key={l.alpha} value={l.alpha}>
                  {`${Math.round(l.confidence_level * 100)}%: answers ${pct(l.answered)}${l.accuracy_answered != null ? ` (${pct(l.accuracy_answered)} correct)` : ''}`}
                </option>
              ))}
            </select>
          </div>
          <p className="hint">“Answers” is the share of ECGs the model decides on; the rest go to a clinician. Measured on {status.n_test.toLocaleString()} held-out test ECGs from unseen patients.</p>
        </div>
        <button className="btn primary" disabled={busy || !canRun} type="submit">{busy ? 'Working…' : 'Analyse ECG'}</button>
      </form>

      <main>
        {err && <div className="err" role="alert">{err}</div>}
        {!res ? (
          <div className="empty">
            <h2>Choose an ECG and press “Analyse ECG”</h2>
            <p>You will get a calibrated probability that the ECG is abnormal, a plain statement of whether the model is sure enough to answer, and a lead-by-lead view of where the model looked.</p>
            <p>The model reads the 12-lead tracing itself; age and sex are shown as a second estimate, because the study found they add little once the ECG is known.</p>
          </div>
        ) : <EcgResult r={res} />}
      </main>
    </div>
  )
}

function EcgResult({ r }) {
  const c = COPY[r.decision]
  const p = Math.round(r.probability * 100)
  const set = r.conformal_set.length === 2 ? 'both readings are plausible' : r.conformal_set.length === 0 ? 'neither reading is plausible (unusual input)'
    : r.conformal_set[0] === 1 ? 'only “abnormal”' : 'only “normal”'
  const rec = r.recorded
  const agree = rec ? (r.decision === 'defer' ? 'The model deferred this one.' : (r.decision === 'positive') === rec.abnormal ? 'The model’s answer matches the reference label.' : 'The model’s answer does not match the reference label.') : ''
  return (
    <>
      <section className="band" aria-live="polite">
        <span className={`pill ${r.decision}`}>{c.pill}</span>
        <h2>{c.title}</h2>
        <p>{c.body}</p>
        <div className="scale" role="img" aria-label={`Estimated probability of an abnormal ECG: ${p} percent`}>
          <div className="track" />
          <div className="marker" style={{ left: `${p}%` }}><span>{p}%</span></div>
          <span className="tick" style={{ left: '0%', transform: 'none' }}>0%</span>
          <span className="tick" style={{ left: '50%' }}>50%</span>
          <span className="tick" style={{ left: '100%', transform: 'translateX(-100%)' }}>100%</span>
        </div>
        <div className="meta">
          At the {Math.round(r.target_coverage * 100)}% confidence level, {set}. With age and sex added the estimate is {pct(r.probability_with_age_sex)}; in the study this did not significantly improve accuracy (see Study results).
        </div>
      </section>
      {rec && (
        <section className="sec ref" aria-label="Reference label">
          <h2>Reference label</h2>
          <p className="sub">Shown after the estimate so you can compare. It comes from the PTB-XL annotations (cardiologist-assigned codes grouped into superclasses).</p>
          <p><b>{rec.abnormal ? 'Abnormal' : 'Normal'}</b>{rec.abnormal ? ` (${rec.superclasses.join(', ')})` : ''}. {agree}</p>
        </section>
      )}
      <EcgChart r={r} />
      <p className="note" style={{ marginTop: 28 }}>{r.disclaimer}</p>
    </>
  )
}
