const W = 1000, H = 54

function Trace({ values, rel }) {
  let lo = Infinity, hi = -Infinity
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v }
  const span = hi - lo || 1
  const pts = values.map((v, i) => `${i},${(H - 4 - ((v - lo) / span) * (H - 8)).toFixed(1)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="trace" role="presentation">
      {rel.map((r, b) => Math.abs(r) >= 0.1 && (
        <rect key={b} x={b * 10} y="0" width="10" height={H}
              style={{ fill: r > 0 ? 'var(--up)' : 'var(--down)', opacity: 0.1 + 0.5 * Math.abs(r) }} />
      ))}
      <polyline points={pts} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

export default function EcgChart({ r }) {
  const ex = r.explanation
  const order = ex.lead_importance.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])
  const top = new Set(order.slice(0, 3).map(([, i]) => i))
  const max = Math.max(...ex.lead_importance, 0.001)
  const d = ex.deletion_check
  return (
    <section className="sec ecg" aria-label="Where the model looked">
      <h2>Where the model looked</h2>
      <p className="sub">Each row is one lead (10 seconds). Shading marks moments that pushed the model toward “abnormal” or toward “normal”; the bar beside the name shows how much of the total attribution that lead carries.</p>
      <div className="legend">
        <span><i style={{ background: 'var(--up)' }} />Pushes toward abnormal</span>
        <span><i style={{ background: 'var(--down)' }} />Pushes toward normal</span>
      </div>
      <div className="leads">
        {r.leads.map((name, i) => (
          <div key={name} className={`lead ${top.has(i) ? 'hot' : ''}`}>
            <div className="lname"><b>{name}</b><span className="ibar"><i style={{ width: `${(ex.lead_importance[i] / max) * 100}%` }} /></span>
              <small>{Math.round(ex.lead_importance[i] * 100)}%</small></div>
            <Trace values={r.signal[i]} rel={ex.relevance[i]} />
          </div>
        ))}
        <div className="lead axis"><div className="lname" />
          <div className="ticks">{[0, 2, 4, 6, 8, 10].map((t) => <span key={t}>{t} s</span>)}</div></div>
      </div>
      <div className="check">
        <h3>Does the explanation matter to the model?</h3>
        <p>
          Flat-lining the three leads ranked highest ({d.flatten.join(', ')}) moves the estimate from {Math.round(d.probability_full * 100)}% to {Math.round(d.probability_without_top_leads * 100)}%.
          Flat-lining three random leads moves it to {Math.round(d.probability_without_random_leads_mean * 100)}% on average (20 draws).
          If the first change is not larger than the second, treat the highlighted leads with caution.
        </p>
        <p className="note">Method: {ex.method}. Attributions add up to the model’s output change within {ex.completeness_gap} logit units. This is a model attribution and has not been reviewed by cardiologists.</p>
      </div>
    </section>
  )
}
