# Prism runtime repin to the cudafast harness baseline

Owner: https://github.com/kortexa-ai/shingi-27b/issues/7

Record the read-only cudafast leaderboard audit as `planning/cudafast-perf.md`. Move the owned Prism runtime pin from d8f26eec76da6d09bb708bcba51ef64b8cd868a3 to 4fbda1292562cc1f23904d5867fb79603737547a, the revision the public Bonsai 2 9B CUDA harness vendors, so later kernel ports apply on the same upstream source.

Keep the quantized device-state correction; the new revision still divides device-state byte lengths by `ggml_element_size`. Retire the SM120 recurrent-column patch; upstream #216 derives the four-column layout for every NVIDIA Ampere or newer GPU. Update the build stamp, the runtime revision reported by the model module, the notice, the README text and the patch tests.

Validate with the engine tests, then an isolated CUDA build on smarty under a separate home. Run the explicit fact controls and the fixed 300-prompt fine pass on the UUID-pinned RTX PRO 6000 with the repository's memory gates and compare answers and timings with the dafeb38 runtime. Do not touch the RTX 4090 production worker or stop any service.

Keep the work on a local private branch. Nothing is pushed to GitHub until Franci decides to publish.
