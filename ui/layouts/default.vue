<template>
  <div class="min-h-screen flex">
    <!-- Sidebar -->
    <aside class="w-52 shrink-0 border-r border-surface-border flex flex-col">
      <div class="px-4 py-5 border-b border-surface-border">
        <p class="text-xs text-slate-500 uppercase tracking-widest">Regent</p>
        <p class="text-base font-semibold text-slate-100 mt-0.5">Model Studio</p>
      </div>

      <nav class="flex-1 px-2 py-3 space-y-0.5">
        <nuxt-link
          v-for="link in links"
          :key="link.to"
          :to="link.to"
          class="flex items-center gap-2.5 px-3 py-2 rounded text-sm text-slate-400
                 hover:text-slate-100 hover:bg-surface-raised transition-colors"
          active-class="text-slate-100 bg-surface-raised border-l-2 border-accent"
        >
          <span class="text-base leading-none">{{ link.icon }}</span>
          {{ link.label }}
        </nuxt-link>
      </nav>

      <!-- Server status dot -->
      <div class="px-4 py-3 border-t border-surface-border flex items-center gap-2">
        <span
          class="w-2 h-2 rounded-full"
          :class="online ? 'bg-green-400' : 'bg-red-500'"
        />
        <span class="text-xs text-slate-500">
          {{ online ? 'Server online' : 'Server offline' }}
        </span>
      </div>
    </aside>

    <!-- Main -->
    <main class="flex-1 overflow-auto">
      <nuxt />
    </main>
  </div>
</template>

<script>
export default {
  data() {
    return {
      online: false,
      links: [
        { to: '/',          icon: '◈',  label: 'Dashboard'  },
        { to: '/pipeline',  icon: '⬡',  label: 'Pipeline'   },
        { to: '/training',  icon: '⟳',  label: 'Training'   },
        { to: '/domain',    icon: '◉',  label: 'Domain'     },
        { to: '/inference', icon: '▶',  label: 'Inference'  },
        { to: '/export',    icon: '↑',  label: 'Export'     },
      ],
    }
  },
  mounted() {
    this.ping()
    setInterval(this.ping, 10000)
  },
  methods: {
    async ping() {
      try {
        await this.$api.health()
        this.online = true
      } catch {
        this.online = false
      }
    },
  },
}
</script>
