<template>
  <div class="p-6 space-y-6 max-w-5xl">
    <h1 class="text-xl font-semibold">Dashboard</h1>

    <!-- Row 1: status + params -->
    <div class="grid grid-cols-3 gap-4">
      <!-- Server health -->
      <div class="card space-y-2">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Server</p>
        <div class="flex items-center gap-2">
          <span
            class="w-2.5 h-2.5 rounded-full"
            :class="health.ok ? 'bg-green-400' : 'bg-red-500'"
          />
          <span class="text-sm">{{ health.ok ? 'Online' : 'Offline' }}</span>
        </div>
        <p class="text-xs text-slate-500">
          Model: {{ health.model_loaded ? 'loaded' : 'not loaded' }}
        </p>
      </div>

      <!-- Parameter count -->
      <div class="card space-y-2">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Parameters</p>
        <p class="text-2xl font-semibold">
          {{ modelInfo.parameters ? modelInfo.parameters.total_millions + 'M' : '—' }}
        </p>
        <p class="text-xs text-slate-500">
          d_model {{ modelInfo.model ? modelInfo.model.d_model : '—' }} ·
          {{ modelInfo.model ? modelInfo.model.n_layer : '—' }} layers
        </p>
      </div>

      <!-- Training status -->
      <div class="card space-y-2">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Training</p>
        <div class="flex items-center gap-2">
          <span
            class="w-2.5 h-2.5 rounded-full"
            :class="trainStatus.running ? 'bg-amber-400 animate-pulse' : 'bg-slate-600'"
          />
          <span class="text-sm capitalize">
            {{ trainStatus.running ? 'Running · ' + trainStatus.phase : 'Idle' }}
          </span>
        </div>
        <p class="text-xs text-slate-500">{{ trainStatus.log_lines }} log lines captured</p>
      </div>
    </div>

    <!-- Row 2: model config detail -->
    <div class="card" v-if="modelInfo.model">
      <p class="text-xs text-slate-500 uppercase tracking-wider mb-3">Model Config</p>
      <div class="grid grid-cols-4 gap-3 text-sm">
        <div v-for="(v, k) in flatConfig" :key="k">
          <p class="text-slate-500 text-xs">{{ k }}</p>
          <p class="text-slate-100">{{ v }}</p>
        </div>
      </div>
    </div>

    <!-- Row 3: checkpoints -->
    <div class="card">
      <p class="text-xs text-slate-500 uppercase tracking-wider mb-3">
        Recent Checkpoints
      </p>
      <div v-if="checkpoints.length === 0" class="text-slate-500 text-sm">
        No checkpoints found in ./checkpoints/
      </div>
      <table v-else class="w-full text-sm">
        <thead>
          <tr class="text-left text-slate-500 text-xs border-b border-surface-border">
            <th class="pb-2 font-normal">Phase</th>
            <th class="pb-2 font-normal">File</th>
            <th class="pb-2 font-normal">Size</th>
            <th class="pb-2 font-normal">Modified</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="ck in checkpoints.slice(0, 8)"
            :key="ck.path"
            class="border-b border-surface-border"
          >
            <td class="py-1.5 text-indigo-400 capitalize">{{ ck.phase }}</td>
            <td class="py-1.5 text-slate-300">{{ ck.file }}</td>
            <td class="py-1.5 text-slate-400">{{ ck.size_mb }} MB</td>
            <td class="py-1.5 text-slate-500">{{ formatTime(ck.mtime) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Row 4: active sessions -->
    <div class="card">
      <p class="text-xs text-slate-500 uppercase tracking-wider mb-3">Active Sessions</p>
      <div v-if="sessions.length === 0" class="text-slate-500 text-sm">No active sessions.</div>
      <div v-else class="space-y-1">
        <div
          v-for="s in sessions"
          :key="s.session_id"
          class="flex items-center justify-between text-sm border-b border-surface-border py-1.5"
        >
          <code class="text-xs text-slate-400">{{ s.session_id.slice(0, 8) }}…</code>
          <span class="text-slate-400">{{ s.token_count }} tokens</span>
          <span class="text-slate-500 text-xs">idle {{ s.idle_seconds }}s</span>
          <button class="btn-danger text-xs py-0.5 px-2" @click="deleteSession(s.session_id)">
            Drop
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      health:      { ok: false, model_loaded: false },
      modelInfo:   {},
      trainStatus: { running: false, phase: null, log_lines: 0 },
      checkpoints: [],
      sessions:    [],
    }
  },
  computed: {
    flatConfig() {
      if (!this.modelInfo.model) return {}
      const m = this.modelInfo.model
      return {
        d_model:    m.d_model,
        n_layer:    m.n_layer,
        vocab_size: m.vocab_size,
        ssm_heads:  m.ssm_n_heads,
        ssm_state:  m.ssm_d_state,
        attn_layers: m.attn_layers?.join(', ') || '—',
        ver_enabled: m.ver_enabled ? 'yes' : 'no',
        epg_max:    m.epg_max_nodes,
      }
    },
  },
  mounted() {
    this.refresh()
    this._timer = setInterval(this.refresh, 8000)
  },
  beforeDestroy() {
    clearInterval(this._timer)
  },
  methods: {
    async refresh() {
      try {
        const h = await this.$api.health()
        this.health = { ok: true, model_loaded: h.model_loaded }
      } catch { this.health = { ok: false, model_loaded: false } }

      try { this.modelInfo   = await this.$api.info()        } catch {}
      try { this.trainStatus = await this.$api.trainStatus() } catch {}
      try {
        const ck = await this.$api.checkpoints()
        this.checkpoints = ck.checkpoints || []
      } catch {}
      try {
        const ss = await this.$api.sessions()
        this.sessions = ss.sessions || []
      } catch {}
    },
    async deleteSession(id) {
      await this.$api.deleteSession(id)
      this.sessions = this.sessions.filter(s => s.session_id !== id)
    },
    formatTime(ts) {
      return new Date(ts * 1000).toLocaleString()
    },
  },
}
</script>
