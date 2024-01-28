import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    manifest: true,
    emptyOutDir: true,
    outDir: '../public',
    assetsDir: 'assets',
    rollupOptions: {
      input: './main.js',
      output: {
        entryFileNames: 'assets/js/[name]-[hash].js',
        chunkFileNames: 'assets/js/[name]-[hash].js',
        assetFileNames: ({ name }) => {
          if (/\.(gif|jpe?g|png|svg)$/.test(name ?? '')) {
            return 'assets/images/[name].[ext]';
          }

          if (/\.css$/.test(name ?? '')) {
            return 'assets/css/[name]-[hash].[ext]';
          }

          return 'assets/[name]-[hash].[ext]';
        },
      }
    }
  },
  server: {
    origin: 'http://localhost:8000',
  }
})
