/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          900: '#07111F', // Deep navy / near-black
          800: '#0D2238', // Navy blue
          700: '#15314f',
        },
        accent: {
          cyan: '#22D3EE',
          teal: '#2DD4BF',
          amber: '#FBBF24',
          coral: '#FB7185',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', "Liberation Mono", "Courier New", 'monospace'],
      }
    },
  },
  plugins: [],
}
