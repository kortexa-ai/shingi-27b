# Inference throughput checks — 2026-10-08

The CUDA worker now queries free GPU memory through a retained NVML handle on every
batch. It does not cache memory readings. The startup inventory still uses nvidia-smi.
Parallel decode timers wait for CUDA completion and allocate each call only to its
active sequences. A sole shared prefix stays on the GPU across question waves;
a freshly evaluated prefix also avoids the first host restore. Rejected preparation
invalidates the device-residency hint before memory is cleared.

Eight requests in flight can supply four GPU sequences more consistently. The
transport queue remains bounded at 64, and its admission window remains 2 ms.
Weights, identity calibration, context size, sequence limit and runtime are unchanged.

## Validation

- 123 CPU tests passed on macOS and Linux. The native build passed with `-O2 -Wall -Wextra`.
- The final 4090 run passed 291 distribution comparisons against
  the previous release, with mean total-variation distance 0.00005521,
  maximum 0.00072924, and 100% winner agreement. Frozen gates remain
  mean <= 0.005, maximum <= 0.05, winner agreement >= 99%.
- Coverage includes unrelated concurrent states, shared text/image prefixes, chunked
  choices, malformed and oversized requests, HTTP recovery, and an oversized prefix
  prepared after a valid cached-prefix peer. The valid peer retained its probabilities.
- Short and long prefix controls verify that per-question decode allocations sum to
  the synchronized exchange total. `decode_ms` and `decode_calls` are repeated
  exchange totals, not independent values to sum across responses.
- Sampled peak allocation was 9769 MiB,
  with 2988 MiB minimum free beside existing 4090 services.
  Only Shingi was stopped; ASR, TTS and Miso retained their PIDs and passed health checks.

The 24-request HTTP workload used 1, 4, 8, 8, 4, 1 callers. Four callers took
1.193 and 1.189 seconds; eight took 1.132 and 1.116 seconds. With four callers,
only 4 of 24 requests were in full four-request exchanges in each trial. With eight
callers, 20 of 24 and 24 of 24 were in full exchanges. The batch histogram is recorded
by the worker, not inferred from response timing. These are small fixed-workload
measurements, not a general throughput guarantee.

Full Mappity application trials on this engine used caller counts 4, 8, 8, 4, 4, 8.
Four callers had median search latency 28.92 s (28.90–29.05) and game latency
4.48 s (4.46–4.49); eight took 29.41 s (29.37–29.51) and 4.59 s (4.57–4.60).
Mappity therefore retains four callers. Full batches are not sufficient evidence of
better application throughput. See the [application measurements](https://github.com/kortexa-ai/mappity/blob/main/experiments/README.md)
for frozen inputs, controls and probability differences.

## Memory checks and prefix reuse

A 100-reading check of the NVML guard on the 4090 had median 0.00394 ms and maximum
0.01695 ms after initialization. The command-based check previously had median
25.58 ms over 20 readings. Both observed the same available memory within 1 MiB.
Driver failures stop serving; unsupported unified-memory reporting uses fresh system
available memory as before.

Eight questions sharing a 1,024-token saved prefix produced these native wall times:

| Prefix state | Previous release | This change |
|---|---:|---:|
| Fresh, median of three controls | 1,254.77 ms | 772.29 ms |
| Cached repeat, one control | 672.25 ms | 430.28 ms |
| Host restore time, fresh | 480.41 ms | <0.01 ms |
| Host restore time, cached repeat | 480.55 ms | 240.26 ms |

This table uses the same 192,566,828-byte snapshot and suffixes. Native timing excludes
Python memory-monitor and HTTP overhead. The reference harness uses the previous
native binary and transport with the current memory helper; its Python wall times
are not a measurement of the complete previous serving stack.

Short text prefixes remain uncached. A separate 19-token experiment needed a
157,556,648-byte recurrent-state snapshot. Even a cache hit made nine questions
slower (422 ms versus 281 ms without that snapshot). Saving short prefixes therefore
failed the performance check. The final implementation preserves complete-block
snapshot boundaries and optimizes useful shared-prefix residency instead.

### Confirmed host transfer path

`Snapshot::state` is a host `std::vector<uint8_t>`. The readout calls
`llama_state_seq_get_data` and `llama_state_seq_set_data`, which select the host
writer/reader with flags zero in pinned Prism revision `d8f26eec`. The host writer
calls `ggml_backend_tensor_get`; the reader calls `ggml_backend_tensor_set`.
The CUDA implementations use `cudaMemcpyDeviceToHost` and `cudaMemcpyHostToDevice`,
respectively, and synchronize the stream after each tensor copy. These are real
GPU/host transfers, not just cache metadata changes. The 192,566,828-byte snapshot
above is about 184 MiB. The measured restore time includes transfers and synchronization;
it does not isolate DDR4 bandwidth, PCIe bandwidth or CPU overhead.

Prism also exposes `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`. Its header explicitly says
that another save for a sequence ID invalidates prior on-device snapshots for that
ID. All current cache entries are saved from sequence 0, so changing the flag alone
would break the four-entry cache. Device snapshot ownership and a bounded VRAM budget
need separate validation. The deployed fix keeps a shared prefix resident within an
exchange; the cross-request LRU still stores snapshots in host RAM.

Source: pinned Prism [state I/O](https://github.com/PrismML-Eng/llama.cpp/blob/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/src/llama-context.cpp#L2674),
[device snapshot contract](https://github.com/PrismML-Eng/llama.cpp/blob/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/include/llama.h#L916),
and [CUDA copies](https://github.com/PrismML-Eng/llama.cpp/blob/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/ggml/src/ggml-cuda/ggml-cuda.cu#L786).

## Reproduction

Use `scripts/benchmark_parallel.py` with explicit model, projector and readout paths,
`--parallel 4 --http --out <directory>`; `--reference <results.json>` enforces the
frozen comparison gates. The script never manages services. For a prior-release
reference, `--baseline-backend <previous backend.py>` selects that transport; omit
`--http` when the prior reference does not have the current HTTP error contract.
Raw inputs, outputs, traces, histograms, memory samples and service restoration
records are retained in the work-unit evidence.

- Fixture SHA-256: `a5b0e1a14618f7515129b41beb25ad73de2a2e1e5dd206e19f94db6be7c55e71`.
- Model SHA-256: `c62ae5b61e458aa70aa2f51ebbd193d50e00d170cb318c60341bb68c6471242c`.
- Projector SHA-256: `6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903`.
- Prism runtime: `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`.
- Readout source SHA-256: `c49c12ad78b199fe48794005313951c1ce94ea1c2cae908264c1d9c64448fa0d`.
- Tested Linux binary SHA-256: `e9fbf8f75255fdf6fd3387e38dbabe14261125d7e7f5f10a9e6dd70b42f4f30e`.

Tracking: [engine](https://github.com/kortexa-ai/shingi-27b/issues/3),
[Mappity client](https://github.com/kortexa-ai/mappity/issues/4),
[production rollout](https://github.com/kortexa-ai/models.server/issues/51).
