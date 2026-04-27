import { useEffect, useMemo, useState } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

function formatNumber(value) {
  if (value === null || value === undefined) return '--'
  if (typeof value !== 'number') return String(value)
  return Number.isInteger(value) ? `${value}` : value.toFixed(2)
}

function App() {
  const [metrics, setMetrics] = useState(null)
  const [evalResult, setEvalResult] = useState(null)
  const [hasRequestedEval, setHasRequestedEval] = useState(false)
  const [playResult, setPlayResult] = useState(null)
  const [hasRequestedPlay, setHasRequestedPlay] = useState(false)
  const [loadingMetrics, setLoadingMetrics] = useState(true)
  const [loadingEval, setLoadingEval] = useState(false)
  const [loadingPlay, setLoadingPlay] = useState(false)
  const [error, setError] = useState('')
  const [plotRefreshKey, setPlotRefreshKey] = useState(Date.now())
  const [playRefreshKey, setPlayRefreshKey] = useState(Date.now())

  const plotUrl = useMemo(
    () => `${API_BASE}/api/plot?t=${plotRefreshKey}`,
    [plotRefreshKey],
  )
  const playGifUrl = useMemo(
    () => `${API_BASE}/api/play/gif?t=${playRefreshKey}`,
    [playRefreshKey],
  )

  const fetchMetrics = async () => {
    setLoadingMetrics(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/api/metrics`)
      if (!res.ok) throw new Error(`metrics request failed: ${res.status}`)
      const data = await res.json()
      setMetrics(data)
    } catch (err) {
      console.error(err)
      setError('无法读取 metrics，请确认后端正在运行并且已完成训练。')
    } finally {
      setLoadingMetrics(false)
    }
  }

  const runEvaluate = async () => {
    setHasRequestedEval(true)
    setEvalResult(null)
    setLoadingEval(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/api/evaluate`, { method: 'POST' })
      if (!res.ok) throw new Error(`evaluate request failed: ${res.status}`)
      const data = await res.json()
      setEvalResult(data)
    } catch (err) {
      console.error(err)
      setError('评估失败，请检查 best_model.pth 是否存在。')
    } finally {
      setLoadingEval(false)
    }
  }

  const runPlay = async () => {
    setHasRequestedPlay(true)
    setPlayResult(null)
    setLoadingPlay(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/api/play`, { method: 'POST' })
      if (!res.ok) throw new Error(`play request failed: ${res.status}`)
      const data = await res.json()
      setPlayResult(data)
      setPlayRefreshKey(Date.now())
    } catch (err) {
      console.error(err)
      setError('播放动画生成失败，请确认 best_model.pth 存在且后端依赖完整。')
    } finally {
      setLoadingPlay(false)
    }
  }

  useEffect(() => {
    fetchMetrics()
  }, [])

  return (
    <main className="dashboard">
      <header className="header">
        <div>
          <h1>CartPole DQN Dashboard</h1>
          <p>FastAPI + React Project</p>
        </div>
        <div className="header-actions">
          <button onClick={fetchMetrics} disabled={loadingMetrics}>
            {loadingMetrics ? 'Loading...' : 'Refresh Metrics'}
          </button>
          <button onClick={runEvaluate} disabled={loadingEval}>
            {loadingEval ? 'Evaluating...' : 'Evaluate Agent'}
          </button>
          <button onClick={runPlay} disabled={loadingPlay}>
            {loadingPlay ? 'Playing...' : 'Play Agent'}
          </button>
          <button onClick={() => setPlotRefreshKey(Date.now())}>Refresh Plot</button>
        </div>
      </header>

      {error && <p className="error">{error}</p>}

      <section className="cards">
        <article className="card">
          <h2>Average Reward</h2>
          <p>{formatNumber(metrics?.best_avg_reward)}</p>
        </article>
        <article className="card">
          <h2>Best Reward</h2>
          <p>{formatNumber(metrics?.best_single_reward)}</p>
        </article>
        <article className="card">
          <h2>Total Episodes</h2>
          <p>{formatNumber(metrics?.total_episodes)}</p>
        </article>
        <article className="card">
          <h2>Early Stopped</h2>
          <p>{metrics?.stopped_early ? 'Yes' : 'No'}</p>
        </article>
      </section>

      <section className="plot-panel">
        <h3>Reward Curve</h3>
        <img src={plotUrl} alt="Reward curve" />
      </section>

      <section className="eval-panel">
        <h3>Evaluation Result</h3>

        {!hasRequestedEval && (
          <p className="eval-placeholder">Click "Evaluate Agent" to run 20-episode evaluation.</p>
        )}

        {loadingEval && (
          <p className="eval-loading">
            Evaluating agent<span className="dot dot-1">.</span>
            <span className="dot dot-2">.</span>
            <span className="dot dot-3">.</span>
          </p>
        )}

        {!loadingEval && evalResult && (
          <>
          <div className="eval-summary">
            <p>
              Average Reward: {formatNumber(evalResult.average_reward)} | Best Reward:{' '}
              {formatNumber(evalResult.best_reward)} | Episodes:{' '}
              {formatNumber(evalResult.evaluation_episodes)}
            </p>
          </div>
          <div className="eval-table-wrap">
            <table className="eval-table">
              <thead>
                <tr>
                  <th>Episode</th>
                  <th>Reward</th>
                </tr>
              </thead>
              <tbody>
                {(evalResult.rewards || []).map((reward, index) => (
                  <tr key={`eval-${index + 1}`}>
                    <td>{index + 1}</td>
                    <td>{formatNumber(reward)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          </>
        )}
      </section>

      <section className="play-panel">
        <h3>Play Animation</h3>

        {!hasRequestedPlay && (
          <p className="play-placeholder">Click "Play Agent" to generate one CartPole run animation.</p>
        )}

        {loadingPlay && (
          <p className="play-loading">
            Generating play animation<span className="dot dot-1">.</span>
            <span className="dot dot-2">.</span>
            <span className="dot dot-3">.</span>
          </p>
        )}

        {!loadingPlay && playResult && (
          <>
            <p className="play-summary">
              Reward: {formatNumber(playResult.reward)} | Steps: {formatNumber(playResult.steps)}
            </p>
            <img className="play-gif" src={playGifUrl} alt="CartPole play animation" />
          </>
        )}
      </section>
    </main>
  )
}

export default App
