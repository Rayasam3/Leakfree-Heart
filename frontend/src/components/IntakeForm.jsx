import { useState } from 'react'

const GROUPS = [
  { title: 'About the patient', fields: ['age', 'sex'] },
  { title: 'Symptoms', fields: ['cp', 'exang'] },
  { title: 'Resting measurements', fields: ['trestbps', 'chol', 'fbs', 'restecg'] },
  { title: 'Exercise test', fields: ['thalach', 'oldpeak', 'slope'] },
  { title: 'Imaging', fields: ['ca', 'thal'] },
]
const REQUIRED = ['age', 'sex', 'cp']

// Illustrative profiles for trying the tool. They are made up, not real patients.
const EXAMPLES = {
  'Example with several warning signs': { age: 62, sex: 1, cp: 3, exang: 1, trestbps: 145, chol: 268, fbs: 1, restecg: 1, thalach: 112, oldpeak: 2.8, slope: 1, ca: 2, thal: 3 },
  'Example with reassuring findings': { age: 44, sex: 0, cp: 1, exang: 0, trestbps: 118, chol: 205, fbs: 0, restecg: 0, thalach: 172, oldpeak: 0.2, slope: 0, ca: 0, thal: 1 },
}

export default function IntakeForm({ meta, onSubmit, busy }) {
  const info = meta.feature_info
  const defaults = Object.fromEntries(Object.entries(info).map(([k, v]) => [k, v.default]))
  const [values, setValues] = useState(defaults)
  const [missing, setMissing] = useState(new Set())

  const setVal = (k, v) => setValues((s) => ({ ...s, [k]: v }))
  const toggleMissing = (k) => setMissing((s) => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n })
  const loadExample = (ex) => { setValues({ ...defaults, ...ex }); setMissing(new Set()) }

  const submit = (e) => {
    e.preventDefault()
    const body = {}
    for (const k of Object.keys(info)) {
      if (missing.has(k)) { body[k] = null; continue }
      const v = values[k]
      body[k] = v === '' || v === null || Number.isNaN(Number(v)) ? null : Number(v)
    }
    onSubmit(body)
  }

  return (
    <form className="intake" onSubmit={submit} aria-label="Patient details">
      <h2>Patient details</h2>
      <p className="hint">Enter the results you have. Anything you mark as not recorded is estimated from typical values.</p>
      <div className="examples">
        {Object.entries(EXAMPLES).map(([label, ex]) => (
          <button type="button" key={label} className="btn" onClick={() => loadExample(ex)}>{label}</button>
        ))}
      </div>
      <button className="btn primary" disabled={busy} type="submit">{busy ? 'Working…' : 'Estimate and explain'}</button>
      {GROUPS.map((g) => (
        <div className="group" key={g.title}>
          <h3>{g.title}</h3>
          {g.fields.map((k) => {
            const f = info[k]
            const optional = !REQUIRED.includes(k)
            const off = missing.has(k)
            const id = `f-${k}`
            return (
              <div className="field" key={k}>
                <label htmlFor={id}>{f.label}</label>
                {f.kind === 'numeric' ? (
                  <input id={id} type="number" disabled={off} value={values[k]} min={f.min} max={f.max}
                         step={f.step ?? 1} onChange={(e) => setVal(k, e.target.value)} />
                ) : (
                  <select id={id} disabled={off} value={values[k]} onChange={(e) => setVal(k, e.target.value)}>
                    {Object.entries(f.options).map(([val, text]) => <option key={val} value={val}>{text}</option>)}
                  </select>
                )}
                {f.unit && f.kind === 'numeric' && <span className="unit">{f.unit}</span>}
                {optional && (
                  <label className="na"><input type="checkbox" checked={off} onChange={() => toggleMissing(k)} /> Not recorded</label>
                )}
              </div>
            )
          })}
        </div>
      ))}
    </form>
  )
}
