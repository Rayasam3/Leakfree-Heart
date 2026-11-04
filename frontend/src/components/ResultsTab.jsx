import { useEffect, useState } from 'react'
import { getResultsIndex, getTable } from '../api.js'

const fmt = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v ?? '')

export default function ResultsTab() {
  const [index, setIndex] = useState(null)
  const [name, setName] = useState('')
  const [table, setTable] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    getResultsIndex().then((i) => {
      setIndex(i)
      const first = i.tables.includes('benchmark_summary') ? 'benchmark_summary' : i.tables[0]
      if (first) setName(first)
    }).catch((e) => setErr(e.message))
  }, [])
  useEffect(() => {
    if (!name) return
    getTable(name).then(setTable).catch((e) => setErr(e.message))
  }, [name])

  if (err) return <div className="err">{err}</div>
  if (!index) return <p className="note">Loading results…</p>
  if (!index.tables.length) {
    return <div className="empty"><h2>No experiment results yet</h2>
      <p>Run run_experiments.bat to produce the tables and figures for the paper. They will appear here.</p></div>
  }
  return (
    <div className="results">
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
        {index.figures.map((f) => <figure key={f} style={{ margin: 0 }}><img src={`/figures/${f}`} alt={f.replace('.png', '').replaceAll('_', ' ')} loading="lazy" /></figure>)}
      </div>
    </div>
  )
}
