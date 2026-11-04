async function request(url, options) {
  let res
  try {
    res = await fetch(url, options)
  } catch {
    throw new Error('Cannot reach the API. Start it with start_api.bat and reload this page.')
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
export const getResultsIndex = () => request('/api/results')
export const getTable = (name) => request(`/api/results/${name}`)
