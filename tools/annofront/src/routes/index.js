import Vue from 'vue'
import Router from 'vue-router'
import HelloAnno from '@/components/HelloAnno'
import WpodAnno from '@/components/WpodAnno'
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
    path: '/ocranno',
    name: 'ocranno',
    component: OcrAnno
   },
 ]
})