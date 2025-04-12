/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: "#ff6f00",
        "accent-light": "#ff9e40",
        "accent-dark": "#c43e00",
      },
    },
  },
  plugins: [],
};
