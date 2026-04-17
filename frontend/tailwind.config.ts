import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0a0a0b",
          800: "#111114",
          700: "#1a1a20",
          600: "#26262e",
          500: "#3a3a45",
          400: "#6a6a78",
          300: "#a0a0ac",
          200: "#d8d8de",
        },
        accent: {
          DEFAULT: "#7c5cff",
          soft: "#a594ff",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
