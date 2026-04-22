import { useEffect, useState } from 'react'

function App() {
  const [message, setMessage] = useState('正在连接后端...')
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('http://127.0.0.1:8000/')
      .then((res) => res.json())
      .then((data) => {
        setMessage(data.message)
      })
      .catch((err) => {
        console.error(err)
        setError('后端连接失败')
      })
  }, [])

  return (
    <div style={{ padding: '40px', fontFamily: 'Arial, sans-serif' }}>
      <h1>CartPole Dashboard</h1>
      <p>前端已启动</p>
      <p>后端返回消息：{message}</p>
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  )
}

export default App