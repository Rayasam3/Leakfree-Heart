const COPY = {
  positive: { pill: 'Higher estimated risk', title: 'The model estimates a higher chance of heart disease',
    body: 'The model is confident enough to commit to this answer. It is an estimate from a small research dataset, not a diagnosis.' },
  negative: { pill: 'Lower estimated risk', title: 'The model estimates a lower chance of heart disease',
    body: 'The model is confident enough to commit to this answer. A lower estimate does not rule out heart disease.' },
  defer: { pill: 'Deferred to a clinician', title: 'The model is not sure enough to answer',
    body: 'Both outcomes remain plausible for this patient, so the model abstains. A clinician should review the case; additional test results may help.' },
}

export default function ResultBand({ r }) {
  const c = COPY[r.decision]
  const pct = Math.round(r.probability * 100)
  const base = Math.round((r.base_value ?? 0) * 100)
  const sets = r.conformal_set.length === 0 ? 'neither outcome is plausible (unusual input)'
    : r.conformal_set.length === 2 ? 'both outcomes are plausible'
    : r.conformal_set[0] === 1 ? 'heart disease only' : 'no heart disease only'
  return (
    <section className="band" aria-live="polite">
      <span className={`pill ${r.decision}`}>{c.pill}</span>
      <h2>{c.title}</h2>
      <p>{c.body}</p>
      <div className="scale" role="img" aria-label={`Estimated probability ${pct} percent; typical patient ${base} percent`}>
        <div className="track" />
        <div className="base" style={{ left: `${base}%` }}><span>Typical patient {base}%</span></div>
        <div className="marker" style={{ left: `${pct}%` }}><span>{pct}%</span></div>
        <span className="tick" style={{ left: '0%', transform: 'none' }}>0%</span>
        <span className="tick" style={{ left: '50%' }}>50%</span>
        <span className="tick" style={{ left: '100%', transform: 'translateX(-100%)' }}>100%</span>
      </div>
      <div className="meta">
        At the {Math.round(r.target_coverage * 100)}% confidence level, {sets}. Probabilities are calibrated, and cases below that confidence are deferred.
      </div>
    </section>
  )
}
