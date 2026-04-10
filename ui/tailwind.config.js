module.exports = {
  content: [
    './pages/**/*.vue',
    './components/**/*.vue',
    './layouts/**/*.vue',
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#0f172a',
          raised: '#1e293b',
          border: '#334155',
        },
        accent: {
          DEFAULT: '#6366f1',
          hover: '#4f46e5',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
