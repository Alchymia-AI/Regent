<template>
  <div class="flex h-screen overflow-hidden">

    <!-- ── Left panel: EPG + Essence ── -->
    <aside class="w-72 shrink-0 border-r border-surface-border flex flex-col overflow-y-auto">
      <div class="px-4 py-3 border-b border-surface-border">
        <p class="text-xs text-slate-500 uppercase tracking-wider">Context</p>
      </div>

      <!-- Essence sliders -->
      <div class="px-4 py-3 border-b border-surface-border space-y-2">
        <p class="text-xs text-slate-400 font-medium mb-2">Essence Vector</p>
        <div v-for="dim in essenceDims" :key="dim.key" class="space-y-0.5">
          <div class="flex justify-between text-xs">
            <span class="text-slate-400">{{ dim.label }}</span>
            <span class="text-slate-500">{{ essence[dim.key].toFixed(1) }}</span>
          </div>
          <input
            type="range"
            v-model.number="essence[dim.key]"
            :min="dim.min" :max="dim.max" :step="dim.step"
            class="w-full accent-indigo-500"
          />
        </div>
      </div>

      <!-- EPG nodes selector -->
      <div class="px-4 py-3 flex-1 space-y-2">
        <div class="flex items-center justify-between">
          <p class="text-xs text-slate-400 font-medium">EPG Nodes</p>
          <span class="text-xs text-slate-500">{{ selectedNodes.length }} selected</span>
        </div>
        <input v-model="nodeSearch" class="input text-xs" placeholder="Filter nodes…" />
        <div class="space-y-1 max-h-64 overflow-y-auto">
          <label
            v-for="n in filteredStoreNodes"
            :key="n._id"
            class="flex items-start gap-2 text-xs text-slate-400 cursor-pointer hover:text-slate-100"
          >
            <input type="checkbox" :value="n._id" v-model="selectedNodeIds" class="mt-0.5 accent-indigo-500" />
            <div class="min-w-0">
              <p class="font-medium text-slate-300 truncate">{{ n.key }}</p>
              <p class="text-slate-500 truncate">{{ n.value }}</p>
            </div>
          </label>
          <p v-if="filteredStoreNodes.length === 0" class="text-slate-600 text-xs text-center py-2">
            Add nodes in the Domain page.
          </p>
        </div>
      </div>

      <!-- Generation settings -->
      <div class="px-4 py-3 border-t border-surface-border space-y-2">
        <p class="text-xs text-slate-400 font-medium">Generation</p>
        <div class="grid grid-cols-2 gap-2">
          <div>
            <p class="text-xs text-slate-500 mb-1">Temperature</p>
            <input v-model.number="genConfig.temperature" type="number" step="0.05" min="0" max="2" class="input text-xs py-1" />
          </div>
          <div>
            <p class="text-xs text-slate-500 mb-1">Max tokens</p>
            <input v-model.number="genConfig.max_tokens" type="number" step="64" min="64" max="4096" class="input text-xs py-1" />
          </div>
        </div>
        <label class="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
          <input type="checkbox" v-model="genConfig.verification" class="accent-indigo-500" />
          Grounding verification
        </label>
        <div>
          <p class="text-xs text-slate-500 mb-1">Grounding threshold</p>
          <input v-model.number="genConfig.grounding_threshold" type="range" min="0.1" max="0.9" step="0.05" class="w-full accent-indigo-500" />
          <p class="text-xs text-slate-500">{{ genConfig.grounding_threshold.toFixed(2) }}</p>
        </div>
      </div>

      <!-- Session -->
      <div class="px-4 py-3 border-t border-surface-border">
        <div class="flex items-center justify-between mb-1">
          <p class="text-xs text-slate-500">Session</p>
          <button class="text-xs text-red-400 hover:text-red-300" @click="newSession">New</button>
        </div>
        <code class="text-xs text-slate-600">{{ sessionId ? sessionId.slice(0,8) + '…' : 'none' }}</code>
      </div>
    </aside>

    <!-- ── Chat area ── -->
    <div class="flex-1 flex flex-col overflow-hidden">

      <!-- Messages -->
      <div ref="chatBox" class="flex-1 overflow-y-auto p-5 space-y-4">
        <div
          v-for="(msg, i) in messages"
          :key="i"
          class="flex gap-3"
          :class="msg.role === 'user' ? 'justify-end' : 'justify-start'"
        >
          <!-- User bubble -->
          <div
            v-if="msg.role === 'user'"
            class="max-w-xl bg-indigo-900 border border-indigo-700 rounded-lg px-4 py-3 text-sm text-slate-100"
          >
            {{ msg.content }}
          </div>

          <!-- Assistant bubble -->
          <div v-else class="max-w-2xl space-y-2">
            <div class="bg-surface-raised border border-surface-border rounded-lg px-4 py-3 text-sm leading-relaxed">
              <!-- Grounding token view -->
              <grounding-tokens
                v-if="msg.token_texts && msg.token_texts.length"
                :tokens="msg.token_texts"
                :scores="msg.grounding_scores"
              />
              <span v-else>{{ msg.content }}</span>
            </div>

            <!-- Stats bar -->
            <div v-if="msg.stats" class="flex flex-wrap gap-3 text-xs text-slate-500 px-1">
              <span>{{ msg.stats.total_tokens }} tokens</span>
              <span>{{ msg.stats.inference_time_ms }}ms</span>
              <span v-if="msg.stats.halted_positions.length">
                <span class="badge-halt">{{ msg.stats.halted_positions.length }} halt(s)</span>
              </span>
              <span>
                mean grounding:
                <span
                  :class="meanGroundingClass(meanGrounding(msg.grounding_scores))"
                  class="font-medium"
                >
                  {{ meanGrounding(msg.grounding_scores).toFixed(2) }}
                </span>
              </span>
            </div>
          </div>
        </div>

        <!-- Typing indicator -->
        <div v-if="generating" class="flex gap-3">
          <div class="bg-surface-raised border border-surface-border rounded-lg px-4 py-3 text-sm text-slate-500 flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" style="animation-delay:0ms" />
            <span class="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" style="animation-delay:150ms" />
            <span class="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" style="animation-delay:300ms" />
          </div>
        </div>

        <div v-if="messages.length === 0" class="text-center text-slate-600 mt-16 text-sm">
          Send a message to begin. Select EPG nodes on the left to ground the response.
        </div>
      </div>

      <!-- Error -->
      <div v-if="error" class="mx-5 mb-2 px-3 py-2 bg-red-950 border border-red-800 rounded text-sm text-red-300">
        {{ error }}
      </div>

      <!-- Input bar -->
      <div class="border-t border-surface-border p-4 flex gap-2">
        <textarea
          ref="input"
          v-model="userInput"
          class="input flex-1 resize-none h-12 py-3"
          placeholder="Type a message…"
          @keydown.enter.exact.prevent="sendMessage"
          @keydown.enter.shift.exact="userInput += '\n'"
          :disabled="generating"
        />
        <button
          class="btn-primary px-5"
          :disabled="!userInput.trim() || generating"
          @click="sendMessage"
        >
          ▶
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import GroundingTokens from '~/components/GroundingTokens.vue'

export default {
  components: { GroundingTokens },

  data() {
    return {
      messages: [],
      userInput: '',
      generating: false,
      error: '',
      sessionId: null,
      selectedNodeIds: [],
      nodeSearch: '',
      genConfig: {
        temperature: 0.7,
        max_tokens: 512,
        verification: true,
        grounding_threshold: 0.4,
      },
      essence: {
        essence_index:          5.0,
        essence_influence:      0.0,
        truth_vs_lie:           0.5,
        civility_vs_unruliness: 0.0,
        good_vs_evil:           0.0,
        curiosity:              0.5,
        self_preservation:      0.3,
      },
      essenceDims: [
        { key: 'essence_index',          label: 'Mood',          min: 0, max: 10, step: 0.1  },
        { key: 'essence_influence',       label: 'Influence',     min: 0, max: 10, step: 0.1  },
        { key: 'truth_vs_lie',           label: 'Truth',         min: -1, max: 1, step: 0.05 },
        { key: 'civility_vs_unruliness', label: 'Civility',      min: -1, max: 1, step: 0.05 },
        { key: 'good_vs_evil',           label: 'Good/Evil',     min: -1, max: 1, step: 0.05 },
        { key: 'curiosity',              label: 'Curiosity',     min: 0, max: 1,  step: 0.05 },
        { key: 'self_preservation',      label: 'Self-preserve', min: 0, max: 1,  step: 0.05 },
      ],
    }
  },

  computed: {
    storeNodes() {
      try {
        return JSON.parse(localStorage.getItem('regent_epg_nodes') || '[]')
      } catch { return [] }
    },
    filteredStoreNodes() {
      const q = this.nodeSearch.toLowerCase()
      return q
        ? this.storeNodes.filter(n =>
            n.key.toLowerCase().includes(q) || n.value.toLowerCase().includes(q)
          )
        : this.storeNodes
    },
    selectedNodes() {
      return this.storeNodes.filter(n => this.selectedNodeIds.includes(n._id))
    },
  },

  methods: {
    async sendMessage() {
      const text = this.userInput.trim()
      if (!text || this.generating) return

      this.messages.push({ role: 'user', content: text })
      this.userInput = ''
      this.error = ''
      this.generating = true
      await this.$nextTick()
      this.scrollChat()

      try {
        const body = {
          messages: this.messages
            .filter(m => m.role === 'user')
            .map(m => ({ role: 'user', content: m.content })),
          ...this.genConfig,
          essence: { ...this.essence },
          epg_nodes: this.selectedNodes.length ? this.selectedNodes.map(n => ({
            key:              n.key,
            value:            n.value,
            confidence:       n.confidence,
            activation:       n.activation,
            valence:          n.valence,
            emotional_weight: n.emotional_weight,
            category:         n.category,
          })) : null,
          session_id: this.sessionId,
        }

        const res = await this.$api.generate(body)
        this.sessionId = res.session_id

        this.messages.push({
          role:            'assistant',
          content:         res.text,
          token_texts:     res.token_texts     || [],
          grounding_scores: res.grounding_scores || [],
          stats: {
            total_tokens:     res.total_tokens,
            inference_time_ms: res.inference_time_ms,
            halted_positions: res.halted_positions || [],
          },
        })
      } catch (e) {
        this.error = e.response?.data?.detail || e.message || 'Request failed'
      } finally {
        this.generating = false
        await this.$nextTick()
        this.scrollChat()
      }
    },

    newSession() {
      this.sessionId = null
      this.messages = []
    },

    scrollChat() {
      if (this.$refs.chatBox) {
        this.$refs.chatBox.scrollTop = this.$refs.chatBox.scrollHeight
      }
    },

    meanGrounding(scores) {
      if (!scores || !scores.length) return 1
      return scores.reduce((a, b) => a + b, 0) / scores.length
    },

    meanGroundingClass(val) {
      if (val > 0.6)  return 'text-green-400'
      if (val > 0.3)  return 'text-amber-400'
      return 'text-red-400'
    },
  },
}
</script>
