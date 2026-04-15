/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class", // Enables manual dark mode toggling using the 'dark' class
  theme: {
    extend: {},
  },
  plugins: [require("tailwindcss-animate")],
};
