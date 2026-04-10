// Vuex store — light state shared across pages.
// EPG nodes are owned by domain.vue and persisted to localStorage.
// Training status and server health are polled directly by pages.
// This store holds only what genuinely needs to be global.

export const state = () => ({
  serverOnline: false,
  modelLoaded:  false,
})

export const mutations = {
  SET_SERVER(state, online)      { state.serverOnline = online },
  SET_MODEL_LOADED(state, loaded) { state.modelLoaded = loaded },
}

export const actions = {
  async checkHealth({ commit }) {
    try {
      const res = await this.$axios.$get('/health')
      commit('SET_SERVER', true)
      commit('SET_MODEL_LOADED', res.model_loaded)
    } catch {
      commit('SET_SERVER', false)
      commit('SET_MODEL_LOADED', false)
    }
  },
}
