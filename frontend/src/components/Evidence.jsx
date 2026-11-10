export default function Evidence({ items, activeId }) {
  return (
    <section className="sec ledger" aria-label="Evidence passages">
      <h2>Evidence the explanation can cite</h2>
      <p className="sub">Passages retrieved for this patient. Passages marked as unchecked summaries were written for this prototype and still need to be verified against their source.</p>
      {items.map((p) => (
        <div id={`ev-${p.id}`} key={p.id} className={`passage ${activeId?.id === p.id ? 'hl' : ''}`}>
          <div className="src"><b>{p.id}</b> {p.source}{p.year ? `, ${p.year}` : ''}
            {!p.verified && <span className="flag">Unchecked summary</span>}</div>
          <p>{p.text}</p>
        </div>
      ))}
    </section>
  )
}
