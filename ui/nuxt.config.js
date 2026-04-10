export default {
  ssr: false,

  head: {
    title: 'Regent Model',
    meta: [
      { charset: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
    ],
  },

  css: ['~/assets/main.css'],

  plugins: ['~/plugins/api.js'],

  buildModules: ['@nuxtjs/tailwindcss'],

  modules: ['@nuxtjs/axios'],

  axios: {
    baseURL: process.env.API_URL || 'http://localhost:8400',
  },

  tailwindcss: {
    cssPath: '~/assets/main.css',
    configPath: 'tailwind.config.js',
  },
}
