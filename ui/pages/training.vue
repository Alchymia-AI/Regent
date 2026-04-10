<template>
  <div class="p-6 space-y-6 max-w-4xl">
    <div class="flex items-center justify-between">
      <h1 class="text-xl font-semibold">Training Pipeline</h1>
      <div class="flex gap-2">
        <button
          class="btn-primary"
          :disabled="status.running || !canStart"
          @click="startTraining"
        >
          {{ status.running ? '⟳ Running…' : '▶ Start' }}
        </button>
        <button
          class="btn-danger"
          :disabled="!status.running"
          @click="stopTraining"
        >
          ■ Stop
        </button>
      </div>
    </div>

    <!-- Config inputs -->
    <div class="card space-y-4">
      <p class="text-xs text-slate-500 uppercase tracking-wider">Run Config</p>
      <div class="grid grid-cols-2 gap-3">
        <div>
          <label class="block text-xs text-slate-400 mb-1">Model config YAML</label>
          <input v-model="form.config" class="input" placeholder="configs/regent_7b.yaml" />
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Start stage</label>
          <select v-model.number="form.start_stage" class="input">
            <option :value="1">Stage 1 — Full pipeline (scrape → train)</option>
            <option :value="2">Stage 2 — Tokenizer onwards (corpus ready)</option>
            <option :value="3">Stage 3 — Prepare tokens onwards</option>
            <option :value="4">Stage 4 — Phase 1 base training only</option>
            <option :value="5">Stage 5 — Phase 2 identity onwards</option>
            <option :value="6">Stage 6 — Phase 3 verification onwards</option>
            <option :value="7">Stage 7 — Phase 4 alignment only</option>
          </select>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Checkpoint directory</label>
          <input v-model="form.checkpoint_dir" class="input" placeholder="checkpoints" />
        </div>
      </div>

      <!-- Data source -->
      <div class="border-t border-surface-border pt-4 space-y-3">
        <p class="text-xs text-slate-400 font-medium">Data Source</p>
        <div class="flex gap-4">
          <label class="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
            <input type="radio" v-model="form.data_mode" value="existing" class="accent-indigo-500" />
            Use existing <code class="text-xs text-slate-500">data/raw/</code>
          </label>
          <label class="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
            <input type="radio" v-model="form.data_mode" value="scrape" class="accent-indigo-500" />
            Scrape from sources
          </label>
          <label class="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
            <input type="radio" v-model="form.data_mode" value="synthetic" class="accent-indigo-500" />
            Synthetic (testing only)
          </label>
        </div>

        <!-- Scrape config path -->
        <div v-if="form.data_mode === 'scrape'">
          <label class="block text-xs text-slate-400 mb-1">Scrape config YAML</label>
          <input
            v-model="form.scrape_config"
            class="input"
            placeholder="pipeline.yaml"
          />
          <p class="text-xs text-slate-600 mt-1">
            Defines sources: local files, URLs, HuggingFace datasets, Regent logs.
            Edit <code>pipeline.yaml</code> in the repo root to configure.
          </p>
        </div>

        <!-- Synthetic notice -->
        <div v-if="form.data_mode === 'synthetic'" class="text-xs text-amber-400 flex items-start gap-2">
          <span>⚠</span>
          <span>
            Synthetic data is for architecture validation only. Model quality from synthetic
            training is not meaningful — use real data for any evaluation.
          </span>
        </div>

        <!-- Existing data notice -->
        <div v-if="form.data_mode === 'existing'" class="text-xs text-slate-500">
          Pipeline will look for <code>data/raw/train.txt</code> and <code>data/raw/val.txt</code>.
          If not found it will error at Stage 1.
        </div>
      </div>
    </div>

    <!-- Phase pipeline visual -->
    <div class="card">
      <p class="text-xs text-slate-500 uppercase tracking-wider mb-4">Pipeline Phases</p>
      <div class="flex items-center gap-2">
        <div
          v-for="(phase, i) in phases"
          :key="phase.key"
          class="flex items-center gap-2 flex-1"
        >
          <div
            class="flex-1 rounded p-3 border text-sm"
            :class="phaseClass(phase.key)"
          >
            <p class="font-medium">{{ phase.label }}</p>
            <p class="text-xs mt-1 opacity-70">{{ phase.desc }}</p>
          </div>
          <span v-if="i < phases.length - 1" class="text-slate-600 text-lg shrink-0">→</span>
        </div>
      </div>
    </div>

    <!-- Status + log -->
    <div class="card space-y-3">
      <div class="flex items-center justify-between">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Live Logs</p>
        <div class="flex gap-2 items-center">
          <span class="text-xs text-slate-500">{{ logs.length }} lines</span>
          <button class="btn-ghost text-xs py-0.5 px-2" @click="clearLogs">Clear</button>
          <button class="btn-ghost text-xs py-0.5 px-2" @click="fetchLogs">Refresh</button>
        </div>
      </div>

      <div
        ref="logBox"
        class="bg-surface rounded border border-surface-border h-64 overflow-y-auto p-3 font-mono text-xs leading-5 text-slate-300 space-y-0.5"
      >
        <div
          v-for="(line, i) in logs"
          :key="i"
          :class="lineClass(line)"
        >{{ line }}</div>
        <div v-if="logs.length === 0" class="text-slate-600">No logs yet. Start a training phase.</div>
      </div>
    </div>

    <!-- Checkpoints -->
    <div class="card">
      <p class="text-xs text-slate-500 uppercase tracking-wider mb-3">Checkpoints</p>
      <div v-if="checkpoints.length === 0" class="text-sm text-slate-500">No checkpoints found.</div>
      <table v-else class="w-full text-sm">
        <thead>
          <tr class="text-left text-slate-500 text-xs border-b border-surface-border">
            <th class="pb-2 font-normal">Phase</th>
            <th class="pb-2 font-normal">File</th>
            <th class="pb-2 font-normal">Size</th>
            <th class="pb-2 font-normal">Saved</th>
            <th class="pb-2 font-normal"></th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="ck in checkpoints"
            :key="ck.path"
            class="border-b border-surface-border hover:bg-surface-raised"
          >
            <td class="py-1.5 text-indigo-400 capitalize">{{ ck.phase }}</td>
            <td class="py-1.5 text-slate-300">{{ ck.file }}</td>
            <td class="py-1.5 text-slate-400">{{ ck.size_mb }} MB</td>
            <td class="py-1.5 text-slate-500 text-xs">{{ formatTime(ck.mtime) }}</td>
            <td class="py-1.5">
              <button
                class="btn-ghost text-xs py-0.5 px-2"
                @click="useCheckpoint(ck)"
              >
                Resume from
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      status: { running: false, start_stage: null, log_lines: 0 },
      logs: [],
      checkpoints: [],
      form: {
        config:         'configs/regent_7b.yaml',
        start_stage:    1,
        checkpoint_dir: 'checkpoints',
        data_mode:      'existing',   // 'existing' | 'scrape' | 'synthetic'
        scrape_config:  'pipeline.yaml',
      },
      phases: [
        { key: 'base',         label: 'Phase 1 · Base',         desc: 'Language modelling, full corpus' },
        { key: 'identity',     label: 'Phase 2 · Identity',     desc: 'SFT on Regent conversations' },
        { key: 'verification', label: 'Phase 3 · Verification', desc: 'Ver Head, frozen backbone' },
        { key: 'alignment',    label: 'Phase 4 · Alignment',    desc: 'DPO against reference copy' },
      ],
      _logPoll: null,
      _statusPoll: null,
    }
  },
  computed: {
    canStart() {
      if (!this.form.config) return false
      if (this.form.data_mode === 'scrape' && !this.form.scrape_config) return false
      return true
    },
    activePhaseKey() {
      if (!this.status.running) return null
      const stageToPhase = { 4: 'base', 5: 'identity', 6: 'verification', 7: 'alignment' }
      return stageToPhase[this.status.start_stage] || null
    },
  },
  mounted() {
    this.fetchStatus()
    this.fetchCheckpoints()
    this.fetchLogs()
    this._statusPoll = setInterval(this.fetchStatus, 3000)
    this._logPoll    = setInterval(() => {
      if (this.status.running) this.fetchLogs()
    }, 2000)
  },
  beforeDestroy() {
    clearInterval(this._statusPoll)
    clearInterval(this._logPoll)
  },
  methods: {
    async startTraining() {
      const body = {
        config:          this.form.config,
        start_stage:     this.form.start_stage,
        checkpoint_dir:  this.form.checkpoint_dir,
        synthetic:       this.form.data_mode === 'synthetic',
        scrape_config:   this.form.data_mode === 'scrape' ? this.form.scrape_config : null,
      }
      await this.$api.trainStart(body)
      await this.fetchStatus()
    },
    async stopTraining() {
      await this.$api.trainStop()
      await this.fetchStatus()
    },
    async fetchStatus() {
      try {
        this.status = await this.$api.trainStatus()
      } catch {}
    },
    async fetchLogs() {
      try {
        const res = await this.$api.trainLogs(0, 500)
        this.logs = res.logs || []
        await this.$nextTick()
        if (this.$refs.logBox) {
          this.$refs.logBox.scrollTop = this.$refs.logBox.scrollHeight
        }
      } catch {}
    },
    async fetchCheckpoints() {
      try {
        const res = await this.$api.checkpoints()
        this.checkpoints = res.checkpoints || []
      } catch {}
    },
    clearLogs() { this.logs = [] },
    useCheckpoint(ck) {
      // Map phase name to the stage that follows it
      const phaseToNextStage = {
        base: 5, identity: 6, verification: 7, alignment: 7,
      }
      this.form.start_stage = phaseToNextStage[ck.phase] ?? 4
      this.form.checkpoint_dir = ck.path.split('/').slice(0, -2).join('/') || 'checkpoints'
    },
    formatTime(ts) {
      return new Date(ts * 1000).toLocaleString()
    },
    phaseClass(key) {
      if (this.activePhaseKey === key) {
        return 'border-amber-500 bg-amber-950 text-amber-200'
      }
      if (this.status.phase === key && !this.status.running) {
        return 'border-green-700 bg-green-950 text-green-300'
      }
      return 'border-surface-border text-slate-400'
    },
    lineClass(line) {
      if (/error|failed|exception/i.test(line)) return 'text-red-400'
      if (/warn/i.test(line))  return 'text-amber-400'
      if (/loss|step|epoch/i.test(line)) return 'text-indigo-300'
      return ''
    },
  },
}
</script>
