async function request(url, options) {
  let res
  try {
    res = await fetch(url, options)
  } catch {
    throw new Error('Cannot reach the API. Start it with:  python -m leakfree_heart serve  and reload this page.')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* keep status text */ }
    throw new Error(detail)
  }
  return res.json()
}

const post = (url, body) =>
  request(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const getHealth = () => request('/api/health')
export const getMeta = () => request('/api/meta')
export const postReport = (patient) => post('/api/report', patient)

// Study results: 'clinical' = UCI heart-disease study, 'ecg' = PTB-XL multimodal study
const base = (study) => (study === 'ecg' ? '/api/ecg/results' : '/api/results')
export const figureBase = (study) => (study === 'ecg' ? '/ecg-figures' : '/figures')
export const getResultsIndex = (study = 'clinical') => request(base(study))
export const getTable = (study, name) => request(`${base(study)}/${name}`)

// ECG (multimodal) model
export const getEcgStatus = () => request('/api/ecg/status')
export const getEcgSamples = () => request('/api/ecg/samples')
export const postEcg = (body) => post('/api/ecg/predict', body)
