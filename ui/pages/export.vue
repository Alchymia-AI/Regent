<template>
  <div class="p-6 space-y-6 max-w-4xl">
    <div class="flex items-center justify-between">
      <h1 class="text-xl font-semibold">Export Model</h1>
      <div class="flex gap-2">
        <button
          class="btn-primary"
          :disabled="exporting || !canExport"
          @click="startExport"
        >
          {{ exporting ? '⟳ Exporting…' : '↑ Export' }}
        </button>
      </div>
    </div>

    <!-- Checkpoint picker -->
    <div class="card space-y-4">
      <p class="text-xs text-slate-500 uppercase tracking-wider">Source Checkpoint</p>

      <div v-if="checkpoints.length === 0" class="text-sm text-slate-500">
        No checkpoints found. Run training first.
      </div>

      <div v-else class="space-y-2">
        <label
          v-for="ck in checkpoints"
          :key="ck.path"
          class="flex items-center gap-3 p-3 rounded border cursor-pointer transition-colors"
          :class="form.checkpoint === ck.path
            ? 'border-accent bg-surface-raised'
            : 'border-surface-border hover:bg-surface-raised'"
        >
          <input
            type="radio"
            :value="ck.path"
            v-model="form.checkpoint"
            class="accent-indigo-500"
          />
          <div class="flex-1 min-w-0">
            <p class="text-sm text-slate-200 truncate">{{ ck.file }}</p>
            <p class="text-xs text-slate-500 mt-0.5">
              <span class="capitalize text-indigo-400">{{ ck.phase }}</span>
              · {{ ck.size_mb }} MB · {{ formatTime(ck.mtime) }}
            </p>
          </div>
          <span
            v-if="form.checkpoint === ck.path"
            class="text-indigo-400 text-sm shrink-0"
          >✓</span>
        </label>
      </div>

      <!-- Model config selector -->
      <div>
        <label class="block text-xs text-slate-400 mb-1">Model config</label>
        <div class="grid grid-cols-2 gap-2">
          <label
            v-for="cfg in modelConfigs"
            :key="cfg.path"
            class="flex items-center gap-2 p-2.5 rounded border cursor-pointer transition-colors text-sm"
            :class="form.config === cfg.path
              ? 'border-accent bg-surface-raised text-slate-200'
              : 'border-surface-border text-slate-400 hover:bg-surface-raised'"
          >
            <input type="radio" :value="cfg.path" v-model="form.config" class="accent-indigo-500" />
            <div>
              <p class="font-medium">{{ cfg.label }}</p>
              <p class="text-xs text-slate-500 mt-0.5">{{ cfg.desc }}</p>
            </div>
          </label>
        </div>
        <input v-model="form.config" class="input mt-2" placeholder="or type path manually…" />
      </div>
      <div>
        <label class="block text-xs text-slate-400 mb-1">
          Tokenizer path
          <span class="text-slate-600 normal-case ml-1">(optional — .model file)</span>
        </label>
        <input v-model="form.tokenizer" class="input" placeholder="tokenizer/regent.model" />
      </div>
    </div>

    <!-- Format selection -->
    <div class="card space-y-4">
      <p class="text-xs text-slate-500 uppercase tracking-wider">Export Formats</p>

      <div class="grid grid-cols-2 gap-3">
        <label
          class="flex items-start gap-3 p-4 rounded border cursor-pointer transition-colors"
          :class="formats.hf ? 'border-accent bg-surface-raised' : 'border-surface-border hover:bg-surface-raised'"
        >
          <input type="checkbox" v-model="formats.hf" class="accent-indigo-500 mt-0.5" />
          <div>
            <p class="text-sm font-medium text-slate-200">HuggingFace</p>
            <p class="text-xs text-slate-500 mt-1 leading-relaxed">
              Writes <code>config.json</code>, <code>configuration_regent.py</code>,
              <code>model.safetensors</code>, tokenizer files, and a model card.
              Load with the Regent MLX library via <code>trust_remote_code</code>.
            </p>
          </div>
        </label>

        <label
          class="flex items-start gap-3 p-4 rounded border cursor-pointer transition-colors"
          :class="formats.vllm ? 'border-accent bg-surface-raised' : 'border-surface-border hover:bg-surface-raised'"
        >
          <input type="checkbox" v-model="formats.vllm" class="accent-indigo-500 mt-0.5" />
          <div>
            <p class="text-sm font-medium text-slate-200">vLLM / Docker</p>
            <p class="text-xs text-slate-500 mt-1 leading-relaxed">
              Generates a <code>Dockerfile</code> and <code>docker-compose.yml</code>
              that serve Regent via the native HTTP server. Compatible with any
              OpenAI-API client pointed at the container.
            </p>
          </div>
        </label>
      </div>

      <!-- Weight dtype -->
      <div class="border-t border-surface-border pt-4">
        <p class="text-xs text-slate-400 font-medium mb-3">Weight Dtype</p>
        <div class="grid grid-cols-3 gap-2">
          <label
            v-for="dt in dtypes"
            :key="dt.value"
            class="flex flex-col gap-1 p-3 rounded border cursor-pointer transition-colors"
            :class="form.dtype === dt.value
              ? 'border-accent bg-surface-raised'
              : 'border-surface-border hover:bg-surface-raised'"
          >
            <div class="flex items-center gap-2">
              <input type="radio" :value="dt.value" v-model="form.dtype" class="accent-indigo-500" />
              <span class="text-sm font-medium text-slate-200">{{ dt.label }}</span>
            </div>
            <p class="text-xs text-slate-500 leading-relaxed pl-4">{{ dt.desc }}</p>
          </label>
        </div>

        <!-- Device recommendation -->
        <div class="mt-3 text-xs rounded border border-surface-border px-3 py-2 flex items-start gap-2"
             :class="deviceNote.cls">
          <span class="shrink-0">{{ deviceNote.icon }}</span>
          <span>{{ deviceNote.text }}</span>
        </div>
      </div>

      <!-- Output dir -->
      <div class="grid grid-cols-2 gap-3 pt-2">
        <div>
          <label class="block text-xs text-slate-400 mb-1">Output directory</label>
          <input v-model="form.output_dir" class="input" placeholder="export/regent-7b" />
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Model name</label>
          <input v-model="form.name" class="input" placeholder="regent-7b" />
        </div>
      </div>
    </div>

    <!-- Model card fields -->
    <div class="card space-y-4">
      <p class="text-xs text-slate-500 uppercase tracking-wider">Model Card</p>
      <div>
        <label class="block text-xs text-slate-400 mb-1">Description</label>
        <textarea
          v-model="form.description"
          class="input h-16 resize-none"
          placeholder="Short description of this fine-tune / checkpoint…"
        />
      </div>
      <div class="grid grid-cols-2 gap-3">
        <div>
          <label class="block text-xs text-slate-400 mb-1">License</label>
          <select v-model="form.license" class="input">
            <option value="apache-2.0">Apache 2.0</option>
            <option value="mit">MIT</option>
            <option value="cc-by-4.0">CC BY 4.0</option>
            <option value="cc-by-nc-4.0">CC BY-NC 4.0</option>
            <option value="openrail">OpenRAIL</option>
            <option value="other">Other</option>
          </select>
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">Tags (comma-separated)</label>
          <input v-model="tagsRaw" class="input" placeholder="mamba,ssm,regent,causal-lm" />
        </div>
      </div>
    </div>

    <!-- HuggingFace Hub push -->
    <div v-if="formats.hf" class="card space-y-4">
      <div class="flex items-center justify-between">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Push to HuggingFace Hub</p>
        <label class="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
          <input type="checkbox" v-model="hubEnabled" class="accent-indigo-500" />
          Enable
        </label>
      </div>

      <div v-if="hubEnabled" class="grid grid-cols-2 gap-3">
        <div>
          <label class="block text-xs text-slate-400 mb-1">Repo ID</label>
          <input v-model="form.hf_repo" class="input" placeholder="username/regent-7b" />
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">HF Token</label>
          <input
            v-model="form.hf_token"
            type="password"
            class="input"
            placeholder="hf_… (or set HF_TOKEN env var)"
          />
        </div>
      </div>
      <p v-if="hubEnabled" class="text-xs text-slate-600">
        Requires <code>pip install huggingface_hub</code> on the server.
      </p>
    </div>

    <!-- Export log -->
    <div class="card space-y-3">
      <div class="flex items-center justify-between">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Export Log</p>
        <div class="flex gap-2 items-center">
          <span
            v-if="status.done && !status.error"
            class="text-xs text-green-400"
          >✓ Done</span>
          <span v-if="status.error" class="text-xs text-red-400">✗ Error</span>
          <button class="btn-ghost text-xs py-0.5 px-2" @click="clearLog">Clear</button>
        </div>
      </div>

      <div
        ref="logBox"
        class="bg-surface rounded border border-surface-border h-56 overflow-y-auto p-3 font-mono text-xs leading-5 text-slate-300 space-y-0.5"
      >
        <div
          v-for="(line, i) in displayLog"
          :key="i"
          :class="logLineClass(line)"
        >{{ line }}</div>
        <div v-if="displayLog.length === 0" class="text-slate-600">
          No export started yet.
        </div>
      </div>
    </div>

    <!-- Post-export commands -->
    <div v-if="status.done && !status.error" class="card space-y-4">
      <p class="text-xs text-slate-500 uppercase tracking-wider">Deployment Commands</p>

      <div v-if="formats.hf" class="space-y-2">
        <p class="text-xs text-slate-400 font-medium">HuggingFace (MLX)</p>
        <pre class="bg-surface rounded border border-surface-border p-3 text-xs text-slate-300 overflow-x-auto">{{ hfLoadSnippet }}</pre>
      </div>

      <div v-if="formats.vllm" class="space-y-2">
        <p class="text-xs text-slate-400 font-medium">Docker / vLLM</p>
        <pre class="bg-surface rounded border border-surface-border p-3 text-xs text-slate-300 overflow-x-auto">{{ dockerSnippet }}</pre>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      checkpoints: [],
      status: { running: false, done: false, error: '', output_dir: '', log: [] },
      displayLog: [],
      exporting: false,
      hubEnabled: false,
      tagsRaw: 'mamba,ssm,regent',
      formats: { hf: true, vllm: false },
      form: {
        checkpoint:  '',
        config:      'configs/regent_7b.yaml',
        tokenizer:   '',
        output_dir:  'export/regent',
        name:        'regent',
        description: '',
        license:     'apache-2.0',
        dtype:       'float32',
        hf_repo:     '',
        hf_token:    '',
      },
      modelConfigs: [
        { path: 'configs/regent_370m.yaml',       label: '370M',      desc: 'Prototype · Apple Silicon' },
        { path: 'configs/regent_1_5b_edge.yaml',  label: '1.5B Edge', desc: 'Edge · Jetson / M-series' },
        { path: 'configs/regent_7b.yaml',          label: '7B',        desc: 'Production · multi-GPU' },
        { path: 'configs/regent_test.yaml',        label: 'Test',      desc: 'Architecture validation' },
      ],
      dtypes: [
        { value: 'float32',  label: 'float32',  desc: 'Full precision. Safe on all hardware. ~2× memory.' },
        { value: 'float16',  label: 'float16',  desc: 'CUDA GPUs. Half memory, fast on Ampere+.' },
        { value: 'bfloat16', label: 'bfloat16', desc: 'CUDA Ampere+ and Apple MPS. Better numeric range than float16.' },
      ],
      _poll: null,
    }
  },
  computed: {
    canExport() {
      return this.form.checkpoint && this.form.config && (this.formats.hf || this.formats.vllm)
    },
    deviceNote() {
      const notes = {
        float32:  { icon: 'ℹ', cls: 'text-slate-400 border-slate-700',
                    text: 'Works on CPU, CUDA, and Apple MPS. Use for maximum compatibility.' },
        float16:  { icon: '⚡', cls: 'text-indigo-300 border-indigo-800',
                    text: 'CUDA GPUs (Ampere and later recommended). Load with torch_dtype=torch.float16.' },
        bfloat16: { icon: '⚡', cls: 'text-indigo-300 border-indigo-800',
                    text: 'CUDA Ampere+ or Apple MPS (model.to("mps")). Best choice for Metal.' },
      }
      return notes[this.form.dtype] || notes.float32
    },
    hfLoadSnippet() {
      const dir   = this.form.output_dir || 'export/regent'
      const dtype = this.form.dtype === 'float32' ? 'torch.float32'
                  : this.form.dtype === 'float16' ? 'torch.float16'
                  : 'torch.bfloat16'
      const device = this.form.dtype === 'bfloat16' ? '"mps"  # or "cuda"'
                   : this.form.dtype === 'float16'  ? '"cuda"'
                   : '"cpu"  # or "cuda" / "mps"'
      return `from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "${dir}",
    trust_remote_code=True,
    torch_dtype=${dtype},
).to(${device})

# Load tokenizer (requires tokenizer file in export dir)
tok = AutoTokenizer.from_pretrained("${dir}", trust_remote_code=True)

# Generate
inputs = tok("Hello, Regent!", return_tensors="pt").to(model.device)
out    = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(out[0]))

# Compile for extra speed on CUDA (PyTorch >= 2.0):
# model = torch.compile(model)`
    },
    dockerSnippet() {
      const dir = this.form.output_dir || 'export/regent'
      return `# Build and run
docker compose -f ${dir}/docker-compose.yml up --build

# Or directly
bash ${dir}/start.sh`
    },
  },
  mounted() {
    this.fetchCheckpoints()
  },
  beforeDestroy() {
    clearInterval(this._poll)
  },
  methods: {
    async fetchCheckpoints() {
      try {
        const res = await this.$api.checkpoints()
        this.checkpoints = res.checkpoints || []
        if (this.checkpoints.length && !this.form.checkpoint) {
          this.form.checkpoint = this.checkpoints[0].path
        }
      } catch {}
    },
    async startExport() {
      const selectedFormats = []
      if (this.formats.hf)   selectedFormats.push('hf')
      if (this.formats.vllm) selectedFormats.push('vllm')

      const body = {
        checkpoint:  this.form.checkpoint,
        config:      this.form.config,
        tokenizer:   this.form.tokenizer || null,
        output_dir:  this.form.output_dir,
        name:        this.form.name,
        description: this.form.description,
        license:     this.form.license,
        tags:        this.tagsRaw.split(',').map(t => t.trim()).filter(Boolean),
        formats:     selectedFormats,
        dtype:       this.form.dtype,
        hf_repo:     this.hubEnabled ? (this.form.hf_repo || null) : null,
        hf_token:    this.hubEnabled ? (this.form.hf_token || null) : null,
      }

      this.displayLog = []
      this.exporting  = true
      this.status     = { running: true, done: false, error: '', output_dir: '', log: [] }

      try {
        await this.$api.exportStart(body)
      } catch (e) {
        this.exporting = false
        this.displayLog.push('Failed to start export: ' + (e.message || e))
        return
      }

      this._poll = setInterval(this.fetchStatus, 1500)
    },
    async fetchStatus() {
      try {
        this.status = await this.$api.exportStatus()
        this.displayLog = this.status.log || []
        await this.$nextTick()
        if (this.$refs.logBox) {
          this.$refs.logBox.scrollTop = this.$refs.logBox.scrollHeight
        }
        if (!this.status.running) {
          this.exporting = false
          clearInterval(this._poll)
          this._poll = null
        }
      } catch {}
    },
    clearLog() {
      this.displayLog = []
    },
    formatTime(ts) {
      return new Date(ts * 1000).toLocaleString()
    },
    logLineClass(line) {
      if (/error|failed|exception/i.test(line)) return 'text-red-400'
      if (/warn/i.test(line))  return 'text-amber-400'
      if (/done|complete|wrote|copied/i.test(line)) return 'text-green-400'
      return ''
    },
  },
}
</script>
