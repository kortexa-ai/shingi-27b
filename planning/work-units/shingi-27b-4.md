# Bounded VRAM cache

Owner: https://github.com/kortexa-ai/shingi-27b/issues/4

Use Prism on-device snapshots with four reusable ownership slots, a conservative byte bound, and pinned-entry protection. Default CUDA to VRAM-only state caching; retain explicit host and off modes. Compare full-prefix short-state reuse against block boundaries with unchanged decision gates. Check cache hits, eviction, mixed requests, invalid inputs, image state and allocation churn. Measure actual Mappity endpoints on 4090 and PRO6000 at several budgets before deciding whether a custom secondary tier is worthwhile.

Develop Shingi on a local branch. Validate CPU tests and frozen GPU controls, then publish one consolidated runtime improvement on public main. Keep intermediate branches local. Store full reports under `~/Desktop/ai reports/shingi-vram-cache-2026-10-08/`.
