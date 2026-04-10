<template>
  <span class="grounding-tokens leading-relaxed">
    <span
      v-for="(tok, i) in tokens"
      :key="i"
      class="relative group cursor-default"
      :style="tokenStyle(scores[i])"
    >{{ tok }}<span
        class="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-1.5 py-0.5
               bg-slate-900 border border-surface-border rounded text-xs text-slate-200
               whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none z-10
               transition-opacity duration-100"
      >{{ scoreLabel(scores[i]) }}</span></span>
  </span>
</template>

<script>
export default {
  name: 'GroundingTokens',
  props: {
    tokens: {
      type: Array,
      required: true,
    },
    scores: {
      type: Array,
      required: true,
    },
  },
  methods: {
    tokenStyle(score) {
      if (score === undefined || score === null) return {}
      const s = Math.max(0, Math.min(1, score))

      let r, g, b, a

      if (s > 0.6) {
        // Flow zone: green tint, opacity proportional to confidence
        const intensity = (s - 0.6) / 0.4
        r = 74;  g = 222; b = 128   // green-400
        a = intensity * 0.25
      } else if (s > 0.3) {
        // Caution zone: amber tint
        const intensity = (s - 0.3) / 0.3
        r = 251; g = 191; b = 36    // amber-400
        a = 0.15 + (1 - intensity) * 0.2
      } else {
        // Halt zone: red tint, stronger the lower the score
        const intensity = s / 0.3
        r = 248; g = 113; b = 113   // red-400
        a = 0.35 + (1 - intensity) * 0.3
      }

      return {
        backgroundColor: `rgba(${r},${g},${b},${a})`,
        borderRadius: '2px',
        padding: '0 1px',
      }
    },

    scoreLabel(score) {
      if (score === undefined || score === null) return 'n/a'
      const s = score.toFixed(3)
      if (score > 0.6)  return `✓ ${s} (flow)`
      if (score > 0.3)  return `⚠ ${s} (caution)`
      return `✕ ${s} (halt)`
    },
  },
}
</script>
