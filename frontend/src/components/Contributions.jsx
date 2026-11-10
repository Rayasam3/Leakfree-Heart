export default function Contributions({ items, highlight, topK = 5 }) {
  const max = Math.max(0.01, ...items.map((i) => Math.abs(i.shap)))
  return (
    <section className="sec" aria-label="What drove the estimate">
      <h2>What drove the estimate</h2>
      <p className="sub">Each bar shows how much a result moved the estimated probability, in percentage points, compared with a typical patient.</p>
      <div className="legend">
        <span><i style={{ background: 'var(--down)' }} />Lowers risk</span>
        <span><i style={{ background: 'var(--up)' }} />Raises risk</span>
      </div>
      {items.map((it, idx) => {
        const w = (Math.abs(it.shap) / max) * 50
        const up = it.shap >= 0
        return (
          <div key={it.feature} className={`contrib ${idx >= topK ? 'dim' : ''} ${highlight === it.feature ? 'hl' : ''}`}>
            <div className="name">{it.label}<small>{it.phrase}</small></div>
            <div className="bar"><div className={`fill ${up ? 'up' : 'down'}`} style={{ width: `${w}%` }} /></div>
            <div className="val">{up ? '+' : '−'}{Math.abs(it.shap * 100).toFixed(1)}</div>
          </div>
        )
      })}
    </section>
  )
}
