# Prism runtime repin to the cudafast harness baseline

Owner: https://github.com/kortexa-ai/shingi-27b/issues/7

Record the read-only cudafast leaderboard audit as `planning/cudafast-perf.md`. Move the owned Prism runtime pin from d8f26eec76da6d09bb708bcba51ef64b8cd868a3 to 4fbda1292562cc1f23904d5867fb79603737547a, the revision the public Bonsai 2 9B CUDA harness vendors, so later kernel ports apply on the same upstream source.

Keep the quantized device-state correction; the new revision still divides device-state byte lengths by `ggml_element_size`. Retire the SM120 recurrent-column patch; upstream #216 derives the four-column layout for every NVIDIA Ampere or newer GPU. Update the build stamp, the runtime revision reported by the model module, the notice, the README text and the patch tests.

Validate with the engine tests, then an isolated CUDA build on smarty under a separate home. Run the explicit fact controls and the fixed 300-prompt fine pass on the UUID-pinned RTX PRO 6000 with the repository's memory gates and compare answers and timings with the dafeb38 runtime. Do not touch the RTX 4090 production worker or stop any service.

Result (2026-10-23): published on main after Franci's review of the measurements; the follow-up kernel port stays on a local branch. Engine tests 134/134 on snappy and smarty. Same-GPU A/B on the 6000 in [results/prism-repin](../../results/prism-repin/REPORT.md): speed unchanged within noise, 16/16 controls in every run, maximum probability difference 0.00299 with zero 0.5 crossings. Not bit-identical to d8f26eec.

RTX 4090 (2026-10-23 05:30 UTC, authorized production block): same four-run A/B, 21.39 s to 20.29 s per round (5.1 % faster), 16/16 controls in every run, maximum probability difference 0.00269 with zero 0.5 crossings. The parallel restore left Shingi below its preload gate; it was restarted by hand in boot order and all four services were healthy at 05:41 UTC. Details in the report.
