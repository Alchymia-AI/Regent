<template>
  <div class="p-6 space-y-6 max-w-5xl">
    <div class="flex items-center justify-between">
      <h1 class="text-xl font-semibold">Domain Knowledge</h1>
      <div class="flex gap-2">
        <button class="btn-ghost" @click="activeTab = 'nodes'" :class="{ 'border-accent text-slate-100': activeTab === 'nodes' }">EPG Nodes</button>
        <button class="btn-ghost" @click="activeTab = 'ingest'" :class="{ 'border-accent text-slate-100': activeTab === 'ingest' }">Ingest</button>
      </div>
    </div>

    <!-- ── EPG Nodes Tab ── -->
    <div v-if="activeTab === 'nodes'" class="space-y-4">
      <!-- Search + add -->
      <div class="flex gap-2">
        <input v-model="search" class="input flex-1" placeholder="Search nodes by key or value…" />
        <button class="btn-primary" @click="openNew">+ Add Node</button>
      </div>

      <!-- Category breakdown -->
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="(count, cat) in categoryCounts"
          :key="cat"
          class="px-2 py-0.5 rounded border border-surface-border text-xs text-slate-400
                 hover:border-accent hover:text-slate-100 transition-colors"
          :class="{ 'border-accent text-slate-100': filterCat === cat }"
          @click="filterCat = filterCat === cat ? '' : cat"
        >
          {{ cat }} ({{ count }})
        </button>
      </div>

      <!-- Node table -->
      <div class="card p-0 overflow-hidden">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-xs text-slate-500 border-b border-surface-border">
              <th class="px-4 py-2.5 font-normal">Key</th>
              <th class="px-4 py-2.5 font-normal">Value</th>
              <th class="px-4 py-2.5 font-normal">Category</th>
              <th class="px-4 py-2.5 font-normal">Conf</th>
              <th class="px-4 py-2.5 font-normal">Act</th>
              <th class="px-4 py-2.5 font-normal">Val</th>
              <th class="px-4 py-2.5 font-normal"></th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="node in filteredNodes"
              :key="node._id"
              class="border-b border-surface-border hover:bg-surface-raised"
            >
              <td class="px-4 py-2 text-slate-200 font-medium">{{ node.key }}</td>
              <td class="px-4 py-2 text-slate-400 max-w-xs truncate">{{ node.value }}</td>
              <td class="px-4 py-2">
                <span class="px-1.5 py-0.5 rounded bg-surface text-xs text-indigo-300 border border-surface-border">
                  {{ node.category }}
                </span>
              </td>
              <td class="px-4 py-2 text-slate-400">
                <div class="w-16 bg-surface rounded-full h-1.5 overflow-hidden">
                  <div class="h-full bg-green-500 rounded-full" :style="{ width: node.confidence * 100 + '%' }" />
                </div>
              </td>
              <td class="px-4 py-2 text-slate-400">
                <div class="w-16 bg-surface rounded-full h-1.5 overflow-hidden">
                  <div class="h-full bg-amber-500 rounded-full" :style="{ width: node.activation * 100 + '%' }" />
                </div>
              </td>
              <td class="px-4 py-2 text-slate-400 text-xs">{{ node.valence.toFixed(2) }}</td>
              <td class="px-4 py-2">
                <div class="flex gap-1">
                  <button class="btn-ghost text-xs py-0.5 px-2" @click="editNode(node)">Edit</button>
                  <button class="btn-danger text-xs py-0.5 px-2" @click="deleteNode(node._id)">✕</button>
                </div>
              </td>
            </tr>
            <tr v-if="filteredNodes.length === 0">
              <td colspan="7" class="px-4 py-6 text-center text-slate-500 text-sm">
                No nodes match your filter.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- ── Ingest Tab ── -->
    <div v-if="activeTab === 'ingest'" class="space-y-4">
      <div class="card space-y-3">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Paste JSON Array of Nodes</p>
        <p class="text-xs text-slate-500">
          Expected format: array of objects with
          <code class="text-indigo-300">key, value, confidence, activation, valence, emotional_weight, category</code>.
        </p>
        <textarea
          v-model="ingestRaw"
          class="input h-48 resize-none font-mono text-xs"
          placeholder='[{"key":"Contract clause 12.3","value":"Limitation of liability capped at annual fees","confidence":0.95,"activation":0.8,"valence":-0.1,"emotional_weight":0.5,"category":"constraint"}]'
        />
        <div class="flex gap-2">
          <button class="btn-ghost" @click="parseIngest">Preview</button>
          <button class="btn-primary" :disabled="ingestParsed.length === 0" @click="importNodes">
            Import {{ ingestParsed.length }} nodes
          </button>
        </div>
        <p v-if="ingestError" class="text-red-400 text-xs">{{ ingestError }}</p>
      </div>

      <!-- Preview table -->
      <div v-if="ingestParsed.length > 0" class="card p-0 overflow-hidden">
        <p class="px-4 py-2.5 text-xs text-slate-500 border-b border-surface-border">
          Preview ({{ ingestParsed.length }} nodes)
        </p>
        <table class="w-full text-xs">
          <thead>
            <tr class="text-left text-slate-500 border-b border-surface-border">
              <th class="px-4 py-2 font-normal">Key</th>
              <th class="px-4 py-2 font-normal">Value</th>
              <th class="px-4 py-2 font-normal">Category</th>
              <th class="px-4 py-2 font-normal">Conf</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(n, i) in ingestParsed"
              :key="i"
              class="border-b border-surface-border"
            >
              <td class="px-4 py-1.5 text-slate-200">{{ n.key }}</td>
              <td class="px-4 py-1.5 text-slate-400 max-w-xs truncate">{{ n.value }}</td>
              <td class="px-4 py-1.5 text-indigo-300">{{ n.category }}</td>
              <td class="px-4 py-1.5 text-slate-400">{{ n.confidence }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Node editor modal -->
    <div
      v-if="modal.open"
      class="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
      @click.self="modal.open = false"
    >
      <div class="bg-surface-raised border border-surface-border rounded-lg p-6 w-[480px] space-y-4">
        <p class="text-base font-semibold">{{ modal.isNew ? 'Add Node' : 'Edit Node' }}</p>

        <div class="grid grid-cols-2 gap-3">
          <div class="col-span-2">
            <label class="block text-xs text-slate-400 mb-1">Key</label>
            <input v-model="modal.node.key" class="input" />
          </div>
          <div class="col-span-2">
            <label class="block text-xs text-slate-400 mb-1">Value</label>
            <textarea v-model="modal.node.value" class="input h-20 resize-none" />
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Category</label>
            <select v-model="modal.node.category" class="input">
              <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Emotional Weight</label>
            <input v-model.number="modal.node.emotional_weight" type="number" step="0.1" min="0" max="1" class="input" />
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Confidence (0–1)</label>
            <input v-model.number="modal.node.confidence" type="range" min="0" max="1" step="0.01" class="w-full" />
            <span class="text-xs text-slate-400">{{ modal.node.confidence.toFixed(2) }}</span>
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Activation (0–1)</label>
            <input v-model.number="modal.node.activation" type="range" min="0" max="1" step="0.01" class="w-full" />
            <span class="text-xs text-slate-400">{{ modal.node.activation.toFixed(2) }}</span>
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Valence (−1 to +1)</label>
            <input v-model.number="modal.node.valence" type="range" min="-1" max="1" step="0.05" class="w-full" />
            <span class="text-xs text-slate-400">{{ modal.node.valence.toFixed(2) }}</span>
          </div>
        </div>

        <div class="flex justify-end gap-2 pt-2">
          <button class="btn-ghost" @click="modal.open = false">Cancel</button>
          <button class="btn-primary" @click="saveNode">Save</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
let _nextId = 1

const EMPTY_NODE = () => ({
  key: '', value: '', category: 'domain',
  confidence: 0.8, activation: 0.5, valence: 0.0, emotional_weight: 0.5,
})

export default {
  data() {
    return {
      activeTab: 'nodes',
      search: '',
      filterCat: '',
      nodes: [],
      modal: { open: false, isNew: true, node: EMPTY_NODE(), editId: null },
      ingestRaw: '',
      ingestParsed: [],
      ingestError: '',
      categories: [
        'identity','belief','capability','experience','goal','domain',
        'relationship','emotional','procedural','episodic','semantic',
        'preference','constraint','meta','other',
      ],
    }
  },
  computed: {
    filteredNodes() {
      let list = this.nodes
      if (this.filterCat)
        list = list.filter(n => n.category === this.filterCat)
      if (this.search)
        list = list.filter(n =>
          n.key.toLowerCase().includes(this.search.toLowerCase()) ||
          n.value.toLowerCase().includes(this.search.toLowerCase())
        )
      return list
    },
    categoryCounts() {
      const counts = {}
      for (const n of this.nodes) {
        counts[n.category] = (counts[n.category] || 0) + 1
      }
      return counts
    },
  },
  mounted() {
    // Load persisted nodes from localStorage
    try {
      const saved = localStorage.getItem('regent_epg_nodes')
      if (saved) this.nodes = JSON.parse(saved)
    } catch {}
  },
  methods: {
    persist() {
      localStorage.setItem('regent_epg_nodes', JSON.stringify(this.nodes))
    },
    openNew() {
      this.modal = { open: true, isNew: true, node: EMPTY_NODE(), editId: null }
    },
    editNode(node) {
      this.modal = {
        open: true, isNew: false,
        node: { ...node },
        editId: node._id,
      }
    },
    saveNode() {
      if (!this.modal.node.key) return
      if (this.modal.isNew) {
        this.nodes.push({ ...this.modal.node, _id: _nextId++ })
      } else {
        const idx = this.nodes.findIndex(n => n._id === this.modal.editId)
        if (idx !== -1) this.nodes.splice(idx, 1, { ...this.modal.node, _id: this.modal.editId })
      }
      this.persist()
      this.modal.open = false
    },
    deleteNode(id) {
      this.nodes = this.nodes.filter(n => n._id !== id)
      this.persist()
    },
    parseIngest() {
      this.ingestError = ''
      this.ingestParsed = []
      try {
        const arr = JSON.parse(this.ingestRaw)
        if (!Array.isArray(arr)) throw new Error('Expected a JSON array')
        this.ingestParsed = arr.map(n => ({
          key:              String(n.key   || ''),
          value:            String(n.value || ''),
          category:         String(n.category || 'domain'),
          confidence:       Number(n.confidence      ?? 0.8),
          activation:       Number(n.activation      ?? 0.5),
          valence:          Number(n.valence          ?? 0.0),
          emotional_weight: Number(n.emotional_weight ?? 0.5),
        }))
      } catch (e) {
        this.ingestError = e.message
      }
    },
    importNodes() {
      for (const n of this.ingestParsed) {
        this.nodes.push({ ...n, _id: _nextId++ })
      }
      this.persist()
      this.ingestParsed = []
      this.ingestRaw = ''
      this.activeTab = 'nodes'
    },
  },
}
</script>
