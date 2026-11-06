import { useEffect, useState } from 'react'
import { getHealth, getMeta, postReport } from './api.js'
import IntakeForm from './components/IntakeForm.jsx'
import ResultBand from './components/ResultBand.jsx'
import Contributions from './components/Contributions.jsx'
import Explanation from './components/Explanation.jsx'
import Evidence from './components/Evidence.jsx'
import ResultsTab from './components/ResultsTab.jsx'
import EcgTab from './components/EcgTab.jsx'

export default function App() {
  const [tab, setTab] = useState('ecg')
  const [meta, setMeta] = useState(null)
  const [health, setHealth] = useState(null)
  const [report, setReport] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [active, setActive] = useState(null)

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setErr(e.message))
    getMeta().then(setMeta).catch((e) => setErr(e.message))
  }, [])

  const run = async (body) => {
    setBusy(true); setErr('')
    try { setReport(await postReport(body)) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <>
      <div className="banner" role="note">Research prototype built on small public datasets. Not a medical device and not for clinical decisions.</div>
      <div className="shell">
        <header className="top">
          <h1 className="brand">LeakFree-Heart<small>Heart models that explain themselves and say “not sure” when they should</small></h1>
          {health && tab === 'patient' && <span className="note">Model: {health.model}. Writer: {health.llm_backend === 'template' ? 'fixed template (no language model connected)' : health.llm_backend}.</span>}
        </header>
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'ecg'} onClick={() => setTab('ecg')}>ECG (12-lead)</button>
          <button role="tab" aria-selected={tab === 'patient'} onClick={() => setTab('patient')}>Clinical data</button>
          <button role="tab" aria-selected={tab === 'results'} onClick={() => setTab('results')}>Study results</button>
        </div>
        {err && <div className="err" role="alert">{err}</div>}
        {tab === 'results' ? <ResultsTab /> : tab === 'ecg' ? <EcgTab /> : (
          <div className="layout">
            {meta ? <IntakeForm meta={meta} onSubmit={run} busy={busy} /> : <p className="note">Connecting to the API…</p>}
            <main>
              {!report ? (
                <div className="empty">
                  <h2>Enter a patient and press “Estimate and explain”</h2>
                  <p>You will get a calibrated probability, a plain statement of whether the model is confident enough to answer, the results that drove the estimate, and an explanation in which every statement is linked to a source.</p>
                  <p>Not sure where to start? Load one of the example profiles on the left.</p>
                </div>
              ) : (
                <>
                  <ResultBand r={report} />
                  <div className="cols">
                    <Contributions items={report.attributions} highlight={active?.feature} />
                    <Explanation r={report} activeId={active} setActive={setActive} />
                  </div>
                  <Evidence items={report.evidence} activeId={active} />
                  <p className="note" style={{ marginTop: 28 }}>{report.disclaimer}</p>
                </>
              )}
            </main>
          </div>
        )}
      </div>
    </>
  )
}
