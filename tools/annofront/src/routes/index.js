import Vue from 'vue'
import Router from 'vue-router'
import Annotool from '@/components/Annotool'
import Help from '@/components/Help'

Vue.use(Router)

export const router = new Router({
 mode: 'history',
 routes: [
   {
     path: '/',
     name: 'annotool',
     component: Annotool
   },
   {
     path: '/help',
     name: 'help',
     component: Help
   }
 ]
})