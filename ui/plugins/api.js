export default ({ $axios, store }, inject) => {
  const api = {
    // --- Inference ---
    health: ()         => $axios.$get('/health'),
    info:   ()         => $axios.$get('/info'),
    generate: (body)   => $axios.$post('/generate', body),
    verify:   (body)   => $axios.$post('/verify', body),

    // --- Sessions ---
    sessions:      ()        => $axios.$get('/sessions'),
    deleteSession: (id)      => $axios.$delete(`/session/${id}`),

    // --- Scraping ---
    scrapeStart:  (body)     => $axios.$post('/scrape/start', body),
    scrapeStop:   ()         => $axios.$post('/scrape/stop'),
    scrapeStatus: ()         => $axios.$get('/scrape/status'),
    scrapePreview:()         => $axios.$get('/scrape/preview'),
    corpusStats:  (dir)      => $axios.$get('/scrape/corpus/stats', { params: { output_dir: dir } }),
    corpusClear:  (dir)      => $axios.$delete('/scrape/corpus', { params: { output_dir: dir } }),

    // --- Training ---
    trainStart:  (body)      => $axios.$post('/train/start', body),
    trainStop:   ()          => $axios.$post('/train/stop'),
    trainStatus: ()          => $axios.$get('/train/status'),
    trainLogs:   (offset, limit) =>
      $axios.$get('/train/logs', { params: { offset, limit } }),

    // --- Checkpoints ---
    checkpoints: () => $axios.$get('/checkpoints'),

    // --- Export ---
    exportStart:  (body) => $axios.$post('/export/start', body),
    exportStatus: ()     => $axios.$get('/export/status'),
  }

  inject('api', api)
}
