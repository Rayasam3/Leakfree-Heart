import { useEffect, useState } from 'react'
import { figureBase, getResultsIndex, getTable } from '../api.js'

const fmt = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v ?? '')
const STUDIES = [
  { id: 'ecg', label: 'ECG study (PTB-XL)', first: 'ecg_benchmark',
    empty: 'Run the ECG study (python -m leakfree_heart run-ecg) to produce these tables and figures.' },
  { id: 'clinical', label: 'Heart-disease study (UCI)', first: 'benchmark_summary',
    empty: 'Run the experiments (python -m leakfree_heart run-experiments) to produce these tables and figures.' },
]

export default function ResultsTab() {
  const [study, setStudy] = useState('ecg')
  const [index, setIndex] = useState(null)
  const [name, setName] = useState('')
  const [table, setTable] = useState(null)
  const [err, setErr] = useState('')
  const cur = STUDIES.find((s) => s.id === study)

  useEffect(() => {
    setIndex(null); setTable(null); setName(''); setErr('')
    getResultsIndex(study).then((i) => {
      setIndex(i)
      const first = i.tables.includes(cur.first) ? cur.first : i.tables[0]
      if (first) setName(first)
    }).catch((e) => setErr(e.message))
  }, [study])
  useEffect(() => {
    if (!name) return
    getTable(study, name).then(setTable).catch((e) => setErr(e.message))
  }, [name])

  return (
    <div className="results">
      <div className="seg" role="group" aria-label="Which study">
        {STUDIES.map((s) => (
          <button key={s.id} type="button" aria-pressed={study === s.id} onClick={() => setStudy(s.id)}>{s.label}</button>
        ))}
      </div>
      {err && <div className="err" role="alert">{err}</div>}
      {!err && !index && <p className="note">Loading results…</p>}
      {index && !index.tables.length && (
        <div className="empty"><h2>No results yet for this study</h2><p>{cur.empty} They will appear here.</p></div>
      )}
      {index && index.tables.length > 0 && (
        <>
          <label htmlFor="tbl">Result table </label>
          <select id="tbl" value={name} onChange={(e) => setName(e.target.value)}>
            {index.tables.map((t) => <option key={t} value={t}>{t.replaceAll('_', ' ')}</option>)}
          </select>
          {table && (
            <div className="tbl"><table>
              <thead><tr>{table.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>{table.rows.map((r, i) => <tr key={i}>{table.columns.map((c) => <td key={c}>{fmt(r[c])}</td>)}</tr>)}</tbody>
            </table></div>
          )}
          <div className="figs">
            {index.figures.map((f) => (
              <figure key={f} style={{ margin: 0 }}>
                <img src={`${figureBase(study)}/${f}`} alt={f.replace('.png', '').replaceAll('_', ' ')} loading="lazy" />
              </figure>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
