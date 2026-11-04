import { useState } from 'react'

function Check({ ok, yes, no }) {
  return ok ? <span className="y">✔ {yes}</span> : <span className="n">✖ {no}</span>
}

export default function Explanation({ r, activeId, setActive }) {
  const e = r.explanation
  const [showWhy, setShowWhy] = useState(false)
  if (!e) return <p className="note">The explanation could not be produced. Check that the language model returns valid JSON.</p>
  const v = r.verification
  const m = v.metrics
  return (
    <section className="sec expl" aria-label="Explanation">
      <h2>Explanation, checked against the model and the guidelines</h2>
      <p className="sub">
        Written by {r.generator === 'template' ? 'a fixed template' : r.generator}. Each statement is tested automatically:
        does it match the model’s own attribution, and does the cited source support it?
      </p>
      <p className="summary">{e.summary}</p>
      {e.claims.map((c, i) => {
        const chk = v.claims[i]
        return (
          <div className="claim" key={i}>
            <p className="text">
              {c.statement}
              {(c.citations || []).map((id) => (
                <button key={id} type="button" className={`cite ${activeId?.id === id ? 'on' : ''}`}
                        onMouseEnter={() => setActive({ id, feature: c.feature })} onMouseLeave={() => setActive(null)}
                        onFocus={() => setActive({ id, feature: c.feature })} onBlur={() => setActive(null)}
                        onClick={() => document.getElementById(`ev-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })}>
                  {id}
                </button>
              ))}
            </p>
            {chk && (
              <div className="checks">
                <Check ok={chk.model_faithful} yes="Matches the model" no="Does not match the model" />
                <Check ok={chk.evidence_grounded} yes="Backed by the cited source" no="Not backed by the cited source" />
              </div>
            )}
          </div>
        )
      })}
      {e.recommendations?.length > 0 && (
        <ul className="recs">
          {e.recommendations.map((x, i) => (
            <li key={i}>{x.text}
              {(x.citations || []).map((id) => (
                <button key={id} type="button" className={`cite ${activeId?.id === id ? 'on' : ''}`}
                        onMouseEnter={() => setActive({ id })} onMouseLeave={() => setActive(null)}
                        onFocus={() => setActive({ id })} onBlur={() => setActive(null)}>{id}</button>
              ))}
            </li>
          ))}
        </ul>
      )}
      <p className="note" style={{ marginTop: 14 }}>
        {Math.round(m.fully_verified_rate * 100)}% of statements passed every check; the explanation covers {Math.round(m.recall_at_k * 100)}% of the model’s top drivers.
        {v.n_repairs > 0 && ` The language model corrected itself ${v.n_repairs} time(s).`}
        {v.n_pruned > 0 && ` ${v.n_pruned} unsupported statement(s) were removed.`}{' '}
        <button type="button" className="cite" onClick={() => setShowWhy(!showWhy)}>{showWhy ? 'Hide check details' : 'Show check details'}</button>
      </p>
      {showWhy && (
        <ul className="note">
          {v.claims.map((c, i) => (
            <li key={i}>{c.feature}: {c.reasons.length ? c.reasons.join('; ') : 'all checks passed'} (support score {c.support})</li>
          ))}
        </ul>
      )}
    </section>
  )
}
