import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The React dev server proxies API calls to FastAPI (port 8000).
const target = 'http://127.0.0.1:8000'
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': target, '/figures': target } },
  preview: { port: 5173, proxy: { '/api': target, '/figures': target } },
})
