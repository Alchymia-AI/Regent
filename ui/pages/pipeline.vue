<template>
  <div class="p-6 max-w-3xl space-y-6">

    <!-- Step header -->
    <div class="flex items-center gap-0">
      <div
        v-for="(s, i) in steps"
        :key="i"
        class="flex items-center"
      >
        <button
          class="flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-colors"
          :class="stepBtnClass(i)"
          :disabled="!canVisit(i)"
          @click="canVisit(i) && (currentStep = i)"
        >
          <span
            class="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold shrink-0"
            :class="stepDotClass(i)"
          >{{ stepDone(i) ? '✓' : i + 1 }}</span>
          <span class="hidden sm:inline">{{ s.label }}</span>
        </button>
        <span v-if="i < steps.length - 1" class="text-slate-700 mx-1 text-xs">—</span>
      </div>
    </div>

    <!-- ══════════════════════════════════════════════════════════════ -->
    <!-- STEP 0 · SOURCES                                              -->
    <!-- ══════════════════════════════════════════════════════════════ -->
    <div v-if="currentStep === 0" class="space-y-5">
      <div class="card space-y-4">
        <div class="flex items-center justify-between">
          <p class="text-sm font-medium">Web URLs</p>
          <span class="text-xs text-slate-500">{{ urlCount }} URL{{ urlCount !== 1 ? 's' : '' }} entered</span>
        </div>
        <textarea
          v-model="urlsRaw"
          class="input h-36 resize-none font-mono text-xs"
          placeholder="https://example.com/docs&#10;https://arxiv.org/abs/2312.00752&#10;https://en.wikipedia.org/wiki/State_space_model"
          spellcheck="false"
        />
        <p class="text-xs text-slate-500">One URL per line. The scraper fetches each page, strips HTML, and splits into sentence-level training documents.</p>
      </div>

      <!-- HuggingFace datasets -->
      <div class="card space-y-3">
        <div class="flex items-center justify-between">
          <p class="text-sm font-medium">HuggingFace Datasets</p>
          <button class="btn-ghost text-xs py-0.5 px-2" @click="addHF">+ Add dataset</button>
        </div>
        <p class="text-xs text-slate-500 -mt-1">Requires <code>pip install datasets</code> in the server environment.</p>
        <div
          v-for="(hf, i) in hfSources"
          :key="i"
          class="grid grid-cols-5 gap-2 items-end border border-surface-border rounded p-2"
        >
          <div class="col-span-2">
            <label class="block text-xs text-slate-400 mb-1">Dataset</label>
            <input v-model="hf.dataset" class="input text-xs py-1" placeholder="wikimedia/wikipedia" />
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Split</label>
            <input v-model="hf.split" class="input text-xs py-1" placeholder="train" />
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Column</label>
            <input v-model="hf.column" class="input text-xs py-1" placeholder="text" />
          </div>
          <div class="flex gap-1 items-end">
            <div class="flex-1">
              <label class="block text-xs text-slate-400 mb-1">Max docs</label>
              <input v-model.number="hf.max_docs" type="number" class="input text-xs py-1" />
            </div>
            <button class="btn-danger text-xs py-1 px-2 mb-0.5 shrink-0" @click="hfSources.splice(i, 1)">✕</button>
          </div>
        </div>
        <div v-if="hfSources.length === 0" class="text-xs text-slate-600">No HuggingFace sources added.</div>
      </div>

      <!-- Threshold + output -->
      <div class="card space-y-3">
        <p class="text-sm font-medium">Requirements</p>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-xs text-slate-400 mb-1">Minimum docs to proceed</label>
            <input v-model.number="minDocs" type="number" min="100" step="100" class="input" />
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Output directory</label>
            <input v-model="outputDir" class="input" placeholder="data/raw" />
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Concurrent fetches</label>
            <input v-model.number="concurrency" type="number" min="1" max="50" class="input" />
          </div>
        </div>
        <p class="text-xs text-slate-500">
          ~{{ Math.round(minDocs * 80 / 1000) }}K estimated tokens at minimum.
          The 370M test config needs ≥ 500 docs; 7B production needs millions.
          Higher concurrency scrapes faster but may trigger rate limits.
        </p>
      </div>

      <!-- Existing corpus warning -->
      <div v-if="existingCorpus.total_docs > 0" class="flex items-start gap-3 px-4 py-3 bg-amber-950 border border-amber-800 rounded text-sm text-amber-200">
        <span>⚠</span>
        <div>
          Existing corpus found: <strong>{{ existingCorpus.total_docs }} docs / {{ existingCorpus.total_mb }} MB</strong>.
          New scrape will overwrite it.
          <button class="underline ml-2 text-amber-300" @click="clearCorpus">Clear it now</button>
        </div>
      </div>

      <div class="flex justify-end">
        <button class="btn-primary" :disabled="totalSources === 0" @click="goScrape">
          Start scraping →
        </button>
      </div>
    </div>

    <!-- ══════════════════════════════════════════════════════════════ -->
    <!-- STEP 1 · SCRAPING                                             -->
    <!-- ══════════════════════════════════════════════════════════════ -->
    <div v-if="currentStep === 1" class="space-y-5">

      <!-- Big doc counter -->
      <div class="card flex items-center gap-6">
        <div class="text-center">
          <p class="text-5xl font-bold tabular-nums" :class="scrape.docs_total >= minDocs ? 'text-green-400' : 'text-slate-100'">
            {{ scrape.docs_total.toLocaleString() }}
          </p>
          <p class="text-xs text-slate-500 mt-1">documents collected</p>
        </div>
        <div class="flex-1 space-y-2">
          <!-- URL progress bar -->
          <div v-if="scrape.urls_total > 0">
            <div class="flex justify-between text-xs text-slate-400 mb-1">
              <span>URLs</span>
              <span>{{ scrape.urls_done }} / {{ scrape.urls_total }}</span>
            </div>
            <div class="w-full bg-surface rounded-full h-2">
              <div
                class="h-2 rounded-full bg-indigo-500 transition-all"
                :style="{ width: urlProgressPct + '%' }"
              />
            </div>
          </div>
          <!-- Threshold bar -->
          <div>
            <div class="flex justify-between text-xs text-slate-400 mb-1">
              <span>Threshold</span>
              <span>{{ Math.min(scrape.docs_total, minDocs) }} / {{ minDocs }}</span>
            </div>
            <div class="w-full bg-surface rounded-full h-2">
              <div
                class="h-2 rounded-full transition-all"
                :class="thresholdMet ? 'bg-green-500' : 'bg-amber-500'"
                :style="{ width: Math.min(100, (scrape.docs_total / minDocs) * 100) + '%' }"
              />
            </div>
          </div>
          <!-- Current URL -->
          <p v-if="scrape.current_url" class="text-xs text-slate-500 truncate font-mono">
            ↳ {{ scrape.current_url }}
          </p>
          <div v-if="!scrape.running && scrape.done" class="flex items-center gap-2 text-green-400 text-xs">
            <span>✓</span> Scraping complete
          </div>
          <div v-if="scrape.running" class="flex items-center gap-2 text-amber-400 text-xs">
            <span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse inline-block" />
            Running…
          </div>
        </div>
        <div class="flex flex-col gap-2">
          <button v-if="scrape.running" class="btn-danger" @click="stopScrape">■ Stop</button>
          <button
            v-if="!scrape.running && thresholdMet"
            class="btn-primary"
            @click="currentStep = 2"
          >Continue →</button>
          <button
            v-if="!scrape.running && scrape.docs_total > 0 && !thresholdMet"
            class="btn-ghost text-xs"
            @click="currentStep = 2"
          >Continue anyway</button>
        </div>
      </div>

      <!-- Live log -->
      <div class="card space-y-2">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Scrape log</p>
        <div class="bg-surface rounded border border-surface-border h-40 overflow-y-auto p-3 font-mono text-xs leading-5 text-slate-400 space-y-0.5">
          <div v-for="(line, i) in scrape.log" :key="i" :class="logLineClass(line)">{{ line }}</div>
          <div v-if="!scrape.log.length" class="text-slate-600">Waiting for scraper…</div>
        </div>
      </div>

      <!-- Sample docs -->
      <div v-if="preview.length > 0" class="card space-y-2">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Sample documents</p>
        <div class="space-y-1.5 max-h-48 overflow-y-auto">
          <div
            v-for="(doc, i) in preview"
            :key="i"
            class="text-xs text-slate-400 border-b border-surface-border pb-1.5 line-clamp-2"
          >{{ doc }}</div>
        </div>
      </div>

      <!-- Error count -->
      <div v-if="scrape.error_count > 0" class="text-xs text-red-400 px-1">
        {{ scrape.error_count }} URL{{ scrape.error_count !== 1 ? 's' : '' }} failed to fetch (skipped).
      </div>
    </div>

    <!-- ══════════════════════════════════════════════════════════════ -->
    <!-- STEP 2 · CORPUS REVIEW                                        -->
    <!-- ══════════════════════════════════════════════════════════════ -->
    <div v-if="currentStep === 2" class="space-y-5">
      <div class="card space-y-4">
        <p class="text-sm font-medium">Corpus Summary</p>
        <div class="grid grid-cols-3 gap-4 text-center">
          <div>
            <p class="text-3xl font-bold text-slate-100">{{ corpus.train_docs.toLocaleString() }}</p>
            <p class="text-xs text-slate-500 mt-1">training documents</p>
          </div>
          <div>
            <p class="text-3xl font-bold text-slate-100">{{ corpus.val_docs.toLocaleString() }}</p>
            <p class="text-xs text-slate-500 mt-1">validation documents</p>
          </div>
          <div>
            <p class="text-3xl font-bold text-slate-100">{{ corpus.total_mb }}</p>
            <p class="text-xs text-slate-500 mt-1">MB on disk</p>
          </div>
        </div>

        <!-- Quality indicator -->
        <div
          class="flex items-start gap-3 rounded px-4 py-3 border text-sm"
          :class="corpusQualityClass"
        >
          <span>{{ corpusQualityIcon }}</span>
          <div>
            <p class="font-medium">{{ corpusQualityLabel }}</p>
            <p class="text-xs opacity-80 mt-0.5">{{ corpusQualityDetail }}</p>
          </div>
        </div>

        <div class="flex justify-between pt-1">
          <button class="btn-ghost text-xs" @click="currentStep = 0">← Add more sources</button>
          <button class="btn-primary" :disabled="corpus.total_docs === 0" @click="goConfig">
            Configure training →
          </button>
        </div>
      </div>

      <!-- Sample preview -->
      <div v-if="preview.length > 0" class="card space-y-2">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Sample documents</p>
        <div class="space-y-2 max-h-56 overflow-y-auto">
          <div
            v-for="(doc, i) in preview.slice(0, 10)"
            :key="i"
            class="text-xs text-slate-400 border-b border-surface-border pb-2 last:border-0"
          >{{ doc }}</div>
        </div>
      </div>
    </div>

    <!-- ══════════════════════════════════════════════════════════════ -->
    <!-- STEP 3 · TRAINING CONFIG                                      -->
    <!-- ══════════════════════════════════════════════════════════════ -->
    <div v-if="currentStep === 3" class="space-y-5">
      <div class="card space-y-4">
        <p class="text-sm font-medium">Training Configuration</p>

        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-xs text-slate-400 mb-1">Model config</label>
            <select v-model="trainForm.config" class="input">
              <option value="configs/regent_370m.yaml">Regent 370M — prototype</option>
              <option value="configs/regent_1_5b_edge.yaml">Regent 1.5B — edge</option>
              <option value="configs/regent_7b.yaml">Regent 7B — production</option>
              <option value="configs/regent_test.yaml">Test config — fast validation</option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Start stage</label>
            <select v-model.number="trainForm.start_stage" class="input">
              <option :value="1">Stage 1 — Full (tokenize → all phases)</option>
              <option :value="2">Stage 2 — Skip scrape (corpus ready)</option>
              <option :value="3">Stage 3 — Tokenizer exists</option>
              <option :value="4">Stage 4 — Phase 1 base only</option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Checkpoint dir</label>
            <input v-model="trainForm.checkpoint_dir" class="input" placeholder="checkpoints" />
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Data output dir</label>
            <input :value="outputDir" class="input" disabled />
          </div>
        </div>

        <!-- Run plan summary -->
        <div class="border border-surface-border rounded p-3 space-y-1 bg-surface">
          <p class="text-xs text-slate-400 uppercase tracking-wider mb-2">Run plan</p>
          <div v-for="(step, i) in runPlan" :key="i" class="flex items-center gap-2 text-xs text-slate-400">
            <span class="text-indigo-400">{{ i + 1 }}.</span> {{ step }}
          </div>
        </div>

        <div class="flex justify-between pt-1">
          <button class="btn-ghost text-xs" @click="currentStep = 2">← Review corpus</button>
          <button class="btn-primary" @click="startTraining">▶ Start training</button>
        </div>
      </div>
    </div>

    <!-- ══════════════════════════════════════════════════════════════ -->
    <!-- STEP 4 · TRAINING                                             -->
    <!-- ══════════════════════════════════════════════════════════════ -->
    <div v-if="currentStep === 4" class="space-y-5">

      <!-- Phase pipeline -->
      <div class="card">
        <div class="flex items-center justify-between mb-4">
          <p class="text-xs text-slate-500 uppercase tracking-wider">Pipeline Phases</p>
          <div class="flex gap-2">
            <button v-if="!trainStatus.running" class="btn-primary text-xs" @click="startTraining">▶ Restart</button>
            <button v-if="trainStatus.running" class="btn-danger text-xs" @click="stopTraining">■ Stop</button>
          </div>
        </div>
        <div class="flex items-center gap-1.5">
          <div
            v-for="(ph, i) in phases"
            :key="ph.key"
            class="flex items-center gap-1.5 flex-1"
          >
            <div class="flex-1 rounded p-2.5 border text-xs" :class="phaseClass(ph.key)">
              <p class="font-medium">{{ ph.label }}</p>
              <p class="opacity-60 mt-0.5">{{ ph.desc }}</p>
            </div>
            <span v-if="i < phases.length - 1" class="text-slate-700 shrink-0">→</span>
          </div>
        </div>
      </div>

      <!-- Checkpoints -->
      <div class="card" v-if="checkpoints.length > 0">
        <p class="text-xs text-slate-500 uppercase tracking-wider mb-3">Checkpoints saved</p>
        <div class="space-y-1">
          <div
            v-for="ck in checkpoints"
            :key="ck.path"
            class="flex items-center justify-between text-xs border-b border-surface-border py-1.5"
          >
            <span class="text-indigo-400 capitalize w-24">{{ ck.phase }}</span>
            <span class="text-slate-300">{{ ck.file }}</span>
            <span class="text-slate-500">{{ ck.size_mb }} MB</span>
            <span class="text-slate-600">{{ formatTime(ck.mtime) }}</span>
          </div>
        </div>
      </div>

      <!-- Live log -->
      <div class="card space-y-2">
        <div class="flex items-center justify-between">
          <p class="text-xs text-slate-500 uppercase tracking-wider">Training log</p>
          <span class="text-xs text-slate-600">{{ trainLogs.length }} lines</span>
        </div>
        <div
          ref="trainLogBox"
          class="bg-surface rounded border border-surface-border h-64 overflow-y-auto p-3 font-mono text-xs leading-5 text-slate-400 space-y-0.5"
        >
          <div v-for="(line, i) in trainLogs" :key="i" :class="logLineClass(line)">{{ line }}</div>
          <div v-if="!trainLogs.length" class="text-slate-600">No output yet.</div>
        </div>
      </div>
    </div>

  </div>
</template>

<script>
export default {
  name: 'PipelinePage',

  data() {
    return {
      currentStep: 0,

      // Step 0: sources
      urlsRaw:     '',
      hfSources:   [],
      minDocs:     1000,
      concurrency: 10,
      outputDir:   'data/raw',
      existingCorpus: { total_docs: 0, total_mb: 0 },

      // Step 1: scraping
      scrape: {
        running: false, done: false,
        urls_total: 0, urls_done: 0,
        current_url: '', docs_total: 0,
        error_count: 0, log: [],
      },
      preview: [],

      // Step 2: corpus
      corpus: { train_docs: 0, val_docs: 0, total_docs: 0, total_mb: 0 },

      // Step 3 + 4: training
      trainForm: {
        config:         'configs/regent_370m.yaml',
        start_stage:    2,
        checkpoint_dir: 'checkpoints',
      },
      trainStatus: { running: false, start_stage: null },
      trainLogs:   [],
      checkpoints: [],

      steps: [
        { label: 'Sources'   },
        { label: 'Scraping'  },
        { label: 'Review'    },
        { label: 'Configure' },
        { label: 'Training'  },
      ],
      phases: [
        { key: 'base',         label: 'Phase 1 · Base',         desc: 'Language modelling' },
        { key: 'identity',     label: 'Phase 2 · Identity',     desc: 'SFT on Regent logs' },
        { key: 'verification', label: 'Phase 3 · Verify',       desc: 'Ver Head training' },
        { key: 'alignment',    label: 'Phase 4 · Align',        desc: 'DPO alignment' },
      ],

      _scrapePoll: null,
      _trainPoll:  null,
    }
  },

  computed: {
    urlList() {
      return this.urlsRaw
        .split('\n')
        .map(u => u.trim())
        .filter(u => u.startsWith('http'))
    },
    urlCount()      { return this.urlList.length },
    totalSources()  { return this.urlList.length + this.hfSources.filter(h => h.dataset).length },
    urlProgressPct() {
      if (!this.scrape.urls_total) return 0
      return Math.round((this.scrape.urls_done / this.scrape.urls_total) * 100)
    },
    thresholdMet()  { return this.scrape.docs_total >= this.minDocs },

    corpusQualityClass() {
      const d = this.corpus.total_docs
      if (d === 0) return 'border-red-800 bg-red-950 text-red-300'
      if (d < this.minDocs) return 'border-amber-800 bg-amber-950 text-amber-200'
      return 'border-green-800 bg-green-950 text-green-300'
    },
    corpusQualityIcon() {
      const d = this.corpus.total_docs
      if (d === 0) return '✕'
      if (d < this.minDocs) return '⚠'
      return '✓'
    },
    corpusQualityLabel() {
      const d = this.corpus.total_docs
      if (d === 0) return 'No corpus found'
      if (d < this.minDocs) return `Below threshold (need ${this.minDocs.toLocaleString()})`
      return 'Corpus meets training threshold'
    },
    corpusQualityDetail() {
      const d = this.corpus.total_docs
      if (d === 0) return 'Go back and add sources to scrape.'
      const est = Math.round(d * 80 / 1000)
      return `~${est.toLocaleString()}K estimated tokens. ${d < this.minDocs ? 'Add more sources to improve model quality.' : 'Ready to proceed.'}`
    },

    runPlan() {
      const plan = []
      const s = this.trainForm.start_stage
      if (s <= 1) plan.push('Scrape corpus (using pipeline.yaml)')
      if (s <= 2) plan.push('Train BPE tokenizer on corpus')
      if (s <= 3) plan.push('Pack tokens into .npy arrays')
      if (s <= 4) plan.push('Phase 1 — Base pre-training')
      plan.push('Phase 2 — Identity SFT')
      plan.push('Phase 3 — Verification head')
      plan.push('Phase 4 — DPO alignment')
      return plan
    },
  },

  mounted() {
    this.checkExistingCorpus()
  },

  beforeDestroy() {
    clearInterval(this._scrapePoll)
    clearInterval(this._trainPoll)
  },

  methods: {
    // ── Step navigation ──────────────────────────────────────────────
    stepDone(i) {
      if (i === 0) return this.scrape.docs_total > 0 || this.corpus.total_docs > 0
      if (i === 1) return this.scrape.done || this.corpus.total_docs > 0
      if (i === 2) return this.corpus.total_docs > 0
      if (i === 3) return this.trainStatus.running || this.trainLogs.length > 0
      return false
    },
    canVisit(i) {
      if (i === 0) return true
      if (i === 1) return this.totalSources > 0 || this.scrape.docs_total > 0
      if (i === 2) return this.scrape.done || this.corpus.total_docs > 0
      if (i === 3) return this.corpus.total_docs > 0 || this.scrape.docs_total > 0
      if (i === 4) return true
      return false
    },
    stepBtnClass(i) {
      if (i === this.currentStep) return 'text-slate-100 bg-surface-raised'
      if (this.stepDone(i)) return 'text-green-400 hover:bg-surface-raised'
      return 'text-slate-500 hover:text-slate-300'
    },
    stepDotClass(i) {
      if (this.stepDone(i)) return 'bg-green-600 text-white'
      if (i === this.currentStep) return 'bg-indigo-600 text-white'
      return 'bg-surface-border text-slate-400'
    },

    // ── Step 0 actions ────────────────────────────────────────────────
    addHF() {
      this.hfSources.push({ dataset: '', split: 'train', column: 'text', max_docs: 5000 })
    },
    async checkExistingCorpus() {
      try {
        this.existingCorpus = await this.$api.corpusStats(this.outputDir)
      } catch {}
    },
    async clearCorpus() {
      await this.$api.corpusClear(this.outputDir)
      this.existingCorpus = { total_docs: 0, total_mb: 0 }
    },
    async goScrape() {
      const sources = []
      if (this.urlList.length) {
        sources.push({ type: 'urls', urls: this.urlList })
      }
      for (const hf of this.hfSources) {
        if (hf.dataset) {
          sources.push({ type: 'huggingface', ...hf })
        }
      }
      this.scrape = { running: false, done: false, urls_total: 0, urls_done: 0,
                      current_url: '', docs_total: 0, error_count: 0, log: [] }
      this.preview = []
      await this.$api.scrapeStart({
        sources,
        output_dir:  this.outputDir,
        val_ratio:   0.05,
        min_docs:    this.minDocs,
        concurrency: this.concurrency,
      })
      this.currentStep = 1
      this._startScrapePoll()
    },

    // ── Step 1 actions ────────────────────────────────────────────────
    _startScrapePoll() {
      clearInterval(this._scrapePoll)
      this._scrapePoll = setInterval(async () => {
        try {
          const s = await this.$api.scrapeStatus()
          this.scrape = { ...s, error_count: s.error_count ?? 0 }
          const p = await this.$api.scrapePreview()
          this.preview = p.docs || []
          if (!s.running) {
            clearInterval(this._scrapePoll)
            if (s.done) await this.refreshCorpus()
          }
        } catch {}
      }, 1000)
    },
    async stopScrape() {
      await this.$api.scrapeStop()
      clearInterval(this._scrapePoll)
      await this.refreshCorpus()
    },
    async refreshCorpus() {
      try {
        this.corpus = await this.$api.corpusStats(this.outputDir)
      } catch {}
    },

    // ── Step 2 actions ────────────────────────────────────────────────
    async goConfig() {
      await this.refreshCorpus()
      // Auto-select stage 2 since corpus is ready (skip re-scraping)
      this.trainForm.start_stage = 2
      this.currentStep = 3
    },

    // ── Step 3 + 4 actions ────────────────────────────────────────────
    async startTraining() {
      await this.$api.trainStart({
        config:         this.trainForm.config,
        start_stage:    this.trainForm.start_stage,
        checkpoint_dir: this.trainForm.checkpoint_dir,
        synthetic:      false,
        scrape_config:  null,
      })
      this.currentStep = 4
      this._startTrainPoll()
    },
    async stopTraining() {
      await this.$api.trainStop()
    },
    _startTrainPoll() {
      clearInterval(this._trainPoll)
      this._trainPoll = setInterval(async () => {
        try {
          this.trainStatus = await this.$api.trainStatus()
          const res = await this.$api.trainLogs(0, 500)
          this.trainLogs = res.logs || []
          const ck = await this.$api.checkpoints()
          this.checkpoints = ck.checkpoints || []
          await this.$nextTick()
          if (this.$refs.trainLogBox) {
            this.$refs.trainLogBox.scrollTop = this.$refs.trainLogBox.scrollHeight
          }
        } catch {}
      }, 2000)
    },

    // ── Utilities ─────────────────────────────────────────────────────
    logLineClass(line) {
      if (/✗|error|failed|exception/i.test(line)) return 'text-red-400'
      if (/warn/i.test(line))   return 'text-amber-400'
      if (/✓|complete/i.test(line)) return 'text-green-400'
      if (/loss|step|epoch/i.test(line)) return 'text-indigo-300'
      return ''
    },
    phaseClass(key) {
      const active = this.trainStatus.running
      const stageMap = { base: 4, identity: 5, verification: 6, alignment: 7 }
      if (active && stageMap[key] === this.trainStatus.start_stage)
        return 'border-amber-500 bg-amber-950 text-amber-200'
      return 'border-surface-border text-slate-400'
    },
    formatTime(ts) {
      return new Date(ts * 1000).toLocaleString()
    },
  },
}
</script>
