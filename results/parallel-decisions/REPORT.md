# Parallel inference checks

Measured 2026-10-08 on an RTX PRO 6000 Blackwell, with the released Shingi 27B v3.2
weights, unchanged identity calibration and pinned Prism runtime `d8f26eec`.
The original worker was release `f26c643`. The machine kept its other services running.

## What changed

One model allocation serves up to four independent sequences. The native worker packs
several sequences into each decode batch, reads each question's complete candidate
logits at its own output index, and admits work within a shared 16,384-token KV pool.
A long request runs with fewer neighbors when necessary. Prefix snapshots are restored
once per shared-prefix group in a wave, then forked on the device. Image encoding remains
serial. Image token counts, rather than spatial positions, govern KV admission.

A bounded Python queue routes concurrent callers through one transport owner. It admits
64 waiting calls, collects up to four callers for at most 2 ms, and preserves each caller's
result or error. Invalid input does not discard healthy peers. Queue overflow returns 503;
native failure and shutdown release waiting callers. The serial path remains available.

## Agreement and recovery

The frozen fixture contains 30 requests and 55 distinct judgments: explicit-fact
Choice/Noul/Score cases, three prefix lengths, a 120-option multi-round choice, and
red/blue image cases. Repeated native and actual HTTP calls expand the comparison to
363 paired answer distributions per slot setting. This is implementation agreement,
not a benchmark of general model accuracy or calibration.

Acceptance limits were set before the GPU runs: mean probability total-variation distance
(TVD) at most 0.005, maximum at most 0.05, and at least 99% winner agreement. Every candidate
probability must be finite, normalized, and present; logical input-token counts must match.

| Sequences | Mean TVD | Maximum TVD | Winner agreement |
|---|---:|---:|---:|
| 1 | 0 | 0 | 363/363 |
| 2 | 0.0000620 | 0.0008912 | 363/363 |
| 4 | 0.0000437 | 0.0005506 | 363/363 |

Cache-disabled, cold-cache, warm-cache and disabled-again controls produced identical
probabilities within each sequence setting. They reused 1,024 prefix tokens when enabled.
Mixed unrelated long/short requests, image input and multi-round choices returned 200;
oversized and schema-invalid requests returned 422. Subsequent inference and health checks
passed. Literal media-marker text without attached images is accepted as ordinary text.
Both `/v1/systemone` and `/v1/decisions` were exercised. No response crossed callers.

The 114 CPU tests cover existing API behavior, concurrent routing, mixed input errors,
worker exit, malformed native output, timeout, bounded queue overload and shutdown.
They pass on macOS and Linux. Parallel Metal inference was not measured; macOS keeps
its serial default.

## Fixed-workload timings

Each cell is the median of three observations. The client-concurrency order was
1, 2, 4, 4, 2, 1, 1, 4, 2. Loading is excluded. The model was warmed once before the
checks. Complete requests and responses were retained with fixture and binary hashes.

Eight short requests through the native decision engine:

| Engine sequences | One caller | Two callers | Four callers |
|---|---:|---:|---:|
| 1 | 0.474 s | 0.468 s | 0.474 s |
| 2 | 0.495 s | 0.371 s | 0.368 s |
| 4 | 0.502 s | 0.381 s | 0.289 s |

Twenty-four fixed requests through a local HTTP server:

| Engine sequences | One caller | Two callers | Four callers |
|---|---:|---:|---:|
| 1 | 2.429 s | 1.431 s | 1.472 s |
| 2 | 2.503 s | 1.616 s | 1.167 s |
| 4 | 2.505 s | 1.528 s | 1.147 s |

At the same four-client concurrency, four engine sequences improved native throughput
by 1.64 times and HTTP throughput by 1.28 times versus serial mode. HTTP concurrency can
also overlap transport and application overhead even with a serial engine. Comparing a
serial HTTP client with four clients would overstate the engine's contribution.
Single-caller short requests are slightly slower in parallel mode. Large shared prefixes
and image encoding can dominate a request; more slots do not guarantee a speedup.

Sampled peak GPU allocation above the existing resident workload was 9,400 / 9,700 /
10,000 MiB at 1 / 2 / 4 sequences, including the vision projector. Four sequences added
about 600 MiB in this test. Sampling every 200 ms can miss brief peaks. Free GPU memory
returned to the exact recorded baseline after each owned worker exited.

## RTX 4090 confirmation

The same checks ran on the RTX 4090 on 2026-10-08. Only the existing Shingi service
was stopped during the measurements. ASR, TTS and Miso stayed healthy with unchanged
PIDs; the 6000 workload and memory baseline did not change. Shingi was restored and
health-checked from the process that owned the test block. No new 4090 Xid was logged.
The operator used 11,536 MiB pre-load and 1,536 MiB inference headroom floors.

| Sequences | Mean TVD | Maximum TVD | Winner agreement |
|---|---:|---:|---:|
| 1 | 0 | 0 | 363/363 |
| 2 | 0.0000900 | 0.0007474 | 363/363 |
| 4 | 0.0000679 | 0.0011639 | 363/363 |

Median fixed-workload times, with four clients in every column:

| Engine sequences | Native: eight requests | HTTP: twenty-four requests | Peak allocation |
|---|---:|---:|---:|
| 1 | 0.636 s | 1.959 s | 9,169 MiB |
| 2 | 0.487 s | 1.614 s | 9,469 MiB |
| 4 | 0.422 s | 1.536 s | 9,769 MiB |

Four sequences improved native throughput by 1.51 times and HTTP throughput by
1.28 times versus serial mode at matched client concurrency. They used about
600 MiB more GPU memory; the lowest sampled free memory was 2,988 MiB. Every worker
released its allocation after exit. The complete observations are in `summary.json`.

The original baseline supplies native answer distributions. Each candidate mode also
passes the mixed HTTP and recovery checks. An initial original-worker HTTP run stopped
at the known literal-media-marker defect and restored the service. The benchmark now
expects that original rejection when `--baseline-backend` is set; candidate modes must
accept the literal text. No candidate correctness gate was relaxed.

## Reproduce

Use the unchanged release model and projector. Pin the target GPU by full UUID and
ensure the documented memory headroom. This script does not manage services. Build the
native executable with `scripts/build.sh`, then run from the repository root:

```sh
uv run --locked python scripts/benchmark_parallel.py \
  --executable /path/to/readout --model /path/to/shingi-27b.gguf \
  --projector /path/to/mmproj.gguf --parallel 1 --http --out artifacts/serial
uv run --locked python scripts/benchmark_parallel.py \
  --executable /path/to/readout --model /path/to/shingi-27b.gguf \
  --projector /path/to/mmproj.gguf --parallel 4 --http \
  --reference artifacts/serial/results.json --out artifacts/parallel
```

Use `--baseline-backend /path/to/released/src/shingi/backend.py` with the released
executable to compare with the original transport and runtime. `--canary` selects eight
short text requests for the first load. Full runs preserve exact inputs, outputs, traces,
model/runtime/binary identities, per-request latency, GPU samples and gate results in
`results.json`; `summary.json` contains the compact measurements from this report.
