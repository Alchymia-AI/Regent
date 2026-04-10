import Vue from 'vue'
import Router from 'vue-router'
import { normalizeURL, decode } from 'ufo'
import { interopDefault } from './utils'
import scrollBehavior from './router.scrollBehavior.js'

const _dd8768d6 = () => interopDefault(import('../pages/domain.vue' /* webpackChunkName: "pages/domain" */))
const _3f094d58 = () => interopDefault(import('../pages/inference.vue' /* webpackChunkName: "pages/inference" */))
const _06fbda6b = () => interopDefault(import('../pages/training.vue' /* webpackChunkName: "pages/training" */))
const _73459dde = () => interopDefault(import('../pages/index.vue' /* webpackChunkName: "pages/index" */))

const emptyFn = () => {}

Vue.use(Router)

export const routerOptions = {
  mode: 'history',
  base: '/',
  linkActiveClass: 'nuxt-link-active',
  linkExactActiveClass: 'nuxt-link-exact-active',
  scrollBehavior,

  routes: [{
    path: "/domain",
    component: _dd8768d6,
    name: "domain"
  }, {
    path: "/inference",
    component: _3f094d58,
    name: "inference"
  }, {
    path: "/training",
    component: _06fbda6b,
    name: "training"
  }, {
    path: "/",
    component: _73459dde,
    name: "index"
  }],

  fallback: false
}

export function createRouter (ssrContext, config) {
  const base = (config._app && config._app.basePath) || routerOptions.base
  const router = new Router({ ...routerOptions, base  })

  // TODO: remove in Nuxt 3
  const originalPush = router.push
  router.push = function push (location, onComplete = emptyFn, onAbort) {
    return originalPush.call(this, location, onComplete, onAbort)
  }

  const resolve = router.resolve.bind(router)
  router.resolve = (to, current, append) => {
    if (typeof to === 'string') {
      to = normalizeURL(to)
    }
    return resolve(to, current, append)
  }

  return router
}
