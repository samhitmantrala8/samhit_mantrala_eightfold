/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#172033",
        line: "#d9dee8",
        mint: "#18a57a",
        coral: "#ef6b5b"
      }
    }
  },
  plugins: []
};

