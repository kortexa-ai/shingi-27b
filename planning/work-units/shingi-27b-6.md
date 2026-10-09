# CUDA kernel investigation on RTX PRO 6000

Owner: https://github.com/kortexa-ai/shingi-27b/issues/6

Use the fixed 300-prompt Mappity fine pass from issue 5. Profile representative PQ2 matrix multiplication and Gated DeltaNet launches after warmup. Separate hardware-counter runs from latency measurements. Keep the existing 450 W power limit and record clocks, temperature, memory and service health.

Build an isolated runtime from pinned Prism revision d8f26eec76da6d09bb708bcba51ef64b8cd868a3 with the existing quantized device-state correction. Test existing recurrent column reuse and smaller PQ2 tiles independently. Compare CPU operator references, explicit positive and negative controls, full-workload probabilities, and repeated timings. Do not treat unlabeled probability agreement as a task-quality evaluation.

Keep experiments on a local private branch. Synchronize code through Git. Keep at least 10 GiB free on the UUID-pinned 6000, clean up owned workers, and verify all existing services. Publish measured evidence and any validated improvement as a consolidated public main commit. Runtime deployment is a separate models.server work unit if a production change is justified.
