import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/predict-video': {
        target: 'http://localhost:8000',
        // SSE must not be buffered by the proxy
      },
      '/predict': 'http://localhost:8000',
    }
  }
})
