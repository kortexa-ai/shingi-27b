# Independent sequence scaling on RTX PRO 6000

Owner: https://github.com/kortexa-ai/shingi-27b/issues/5

Measure 4, 8, 16 and 32 independent sequences with the same 300 frozen Mappity place prompts. Keep each place in its own state. Use a 1024-token batch control, then a batch capacity scaled with sequence count to distinguish slot count from token capacity. Record native decode, input preparation, state reset, output extraction, occupancy, GPU memory and probability changes. Profile selected cases with CUDA activity capture if available.

Develop only on a local experimental branch. Start with a small canary, keep at least 10 GiB of free memory on the 6000, stop owned children on failure and verify all existing services. The user authorizes both GPUs, but this trial uses only the 6000. Preserve the production engine and weights. Publish measured results and reproducible experiment material after validation; experimental limits do not become production defaults without evidence.
