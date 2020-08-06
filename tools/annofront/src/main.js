import Vue from 'vue'
import App from './App.vue'
import {router}  from './routes/index.js'
import axios from 'axios'
import vuetify from './plugins/vuetify'
import 'material-design-icons-iconfont/dist/material-design-icons.css'

Vue.config.productionTip = false
Vue.prototype.$http = axios;

new Vue({
  render: h => h(App),
  vuetify,
  router
}).$mount('#app')
