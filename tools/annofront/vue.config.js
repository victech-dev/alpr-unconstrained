module.exports = {
  "devServer": {
    "proxy": {
      "/wpodannos": {
        "target": "http://localhost:3000/wpodannos",
        "changeOrigin": true,
        "pathRewrite": {
          "^/wpodannos": ""
        }
      },
      "/ocrannos": {
        "target": "http://localhost:3000/ocrannos",
        "changeOrigin": true,
        "pathRewrite": {
          "^/ocrannos": ""
        }
      }
    },
    "watchOptions": {
        poll: true
    }
  },
  "outputDir": "../annoback/public",
  "transpileDependencies": [
    "vuetify"
  ]
}