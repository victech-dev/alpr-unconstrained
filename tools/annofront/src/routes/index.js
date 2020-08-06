import Vue from 'vue'
import Router from 'vue-router'
import HelloAnno from '@/components/HelloAnno'
import WpodAnno from '@/components/WpodAnno'
import WpodHelp from '@/components/WpodHelp'
import OcrAnno from '@/components/OcrAnno'

Vue.use(Router)

export const router = new Router({
 mode: 'history',
 routes: [
   {
     path: '/',
     name: 'helloanno',
     component: HelloAnno
   },
   {
    path: '/wpodanno',
    name: 'wpodanno',
    component: WpodAnno
   },
   {
    path: '/wpodhelp',
    name: 'wpodhelp',
    component: WpodHelp
  },
   {
    path: '/ocranno',
    name: 'ocranno',
    component: OcrAnno
   },
 ]
})