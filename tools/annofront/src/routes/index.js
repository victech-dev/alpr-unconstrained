import Vue from 'vue'
import Router from 'vue-router'
import WpodAnno from '@/components/WpodAnno'
import OcrAnno from '@/components/OcrAnno'
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
    path: '/ocranno',
    name: 'ocranno',
    component: OcrAnno
   },
   {
     path: '/help',
     name: 'help',
     component: Help
   }
 ]
})