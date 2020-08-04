import Vue from 'vue'
import Router from 'vue-router'
import WpodAnno from '@/components/WpodAnno'
import Help from '@/components/Help'

Vue.use(Router)

export const router = new Router({
 mode: 'history',
 routes: [
   {
     path: '/',
     name: 'wpodanno',
     component: WpodAnno
   },
   {
     path: '/help',
     name: 'help',
     component: Help
   }
 ]
})