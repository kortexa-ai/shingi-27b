# Bounded device prefix state

## Scope and method

The model weights, vision projector and calibration are unchanged. The runtime
base is Prism `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`. Tests use the same
frozen native/HTTP fixture as the previous parallel release. Mappity tests use
the actual search and game endpoints with frozen OSM data. Each place keeps its
own prompt. Latency includes the application and HTTP transport.

Full operator records, inputs, outputs, binary hashes and service health checks
are retained under `~/Desktop/ai reports/shingi-vram-cache-2026-10-08/`.

## Device-state correctness

The first test of unmodified Prism failed a warm short-prefix replay. Its
probability error reached about 0.499 TVD and changed the winning answer. Cold
evaluation was correct. The device writer and reader divided byte counts by
`ggml_element_size`, which is a quantization-block size for Q8_0. Their tensor
views therefore covered only one thirty-second of the intended elements.

The managed build patch multiplies each block count by `ggml_blck_size`. It
checks the complete pinned source and refuses unknown edits. A capability
symbol makes the worker reject VRAM mode with an uncorrected library. Host
serialization is unchanged. The patch applies only to Shingi's owned checkout;
shared model runtimes are not modified.

The corrected cache-only implementation (`f49de20`, binary SHA-256
`c64d88a63efdfe0473671dafc43bb9f8faf5cbb2e6519036730e7176033378ab`)
passed these RTX PRO 6000 controls:

| Control | Comparisons | Winner agreement | Maximum TVD |
|---|---:|---:|---:|
| Frozen native and HTTP release fixture | 291 | 100% | 0.000952 |
| 1 GiB cache: replay, eviction, pinned peers, invalid peer | 405 | 100% | 0.000350 |
| 256 MiB cache: same controls, including capacity bypass | 405 | 100% | 0.000641 |

The tests cover six prefix sizes, three eviction cycles, concurrent requests
with different states, and replay after an input error. Text snapshot payloads
remain on the device. Host cache keys and state metadata stayed below 1 MiB in
these text controls.

## Memory bound and limitations

There are four ownership slots. Active requests pin their snapshots. Each
allocation is charged by its host serialization size, rounded up to 4 MiB,
plus a 4 MiB alignment allowance. The charge includes slots whose logical cache
entry has expired, because Prism retains those GPU buffers until replacement
or context destruction. Replacement releases the old allocation before it
allocates a different-sized buffer.

The largest reservations observed in the cache-only churn tests were 960 MiB
with a 1024 MiB limit and 256 MiB with a 256 MiB limit. Total device use rose by
1076 MiB and 390 MiB respectively above the model-loaded measurements. Those
total changes also include compute/graph working allocations; the cache limit
is not a cap on total process VRAM. Startup admission reserves cache capacity
in addition to the existing model and headroom floors.

When no eligible slot fits, VRAM mode recomputes the prompt. It does not copy
tensor state to host memory. A 9032-token, nine-question control took about
2.96 s with a fitting snapshot and 28.31 s with the 256 MiB fallback. Small
budgets are safe but can be expensive for large shared contexts.

`--prefix-cache host` retains explicit host snapshots. `--prefix-cache off`
disables snapshots. Neither mode is an automatic secondary tier. The current
cache matches complete prefixes and images exactly; it does not search for
partial common prefixes. Single-question requests use direct inference.

## Parallel batches and final checks

The parallel token batch grows from 512 to 1024. For unequal sequence lengths,
the worker can append at most 128 discarded tokens per decode call, subject to
remaining context capacity. It reads logits at each real prompt end. Prefix
snapshots are taken before these question tails and never contain padding.
Serial mode retains 512-token calls. Mixed-length controls exercise padding in
both request orders and a wide-length batch that falls back without padding.

The final native source SHA-256 is
`b3cbb3eb9c659bbd45c77742eaa3f607f93b4b91575894a0aabbfd67f627c55d`;
the tested executable SHA-256 is
`1a86510f167b59891e181f638c1b7b12f83cd7afbe46c3262073d97bca8282c7`.
On the 4090 it passed all 291 frozen comparisons with 100% winner agreement,
mean TVD 0.00007285 and maximum TVD 0.000611. Both cache churn budgets passed
405 comparisons each; maximum TVD was 0.000433 at 1 GiB and 0.000558 at 256 MiB.
The final accounting also preserves the old allocation charge if a snapshot
replacement fails, and reports zero reused tokens when an image snapshot is
bypassed. The matched off-mode image control checks that last case directly.

The 6000 batching implementation also passed all 291 comparisons in each of
VRAM, host and off modes, with maximum TVD 0.000589, 0.001482 and 0.000703.
All winners agreed. CPU tests passed: 134 engine tests, 18 model integration
checks and 22 Mappity tests. The native build used `-O2 -Wall -Wextra`.
The final identical binary was then checked again on the 6000: 291 comparisons,
100% winner agreement, mean TVD 0.0001070 and maximum TVD 0.001482.

On the 4090, the model-loaded free memory was 2622 MiB. The full fixture reached
1528 MiB free; cache churn stayed above the 1024 MiB serving floor. The larger
batch adds about 534 MiB of peak working storage in the 6000 fixture. Cache
capacity is a separate reservation. Neighboring services retained their PIDs
and passed health checks; each test block restored production Shingi.

## Matched state transfer

The final 4090 executable evaluated the same 1041-token prefix and eight
questions in host, VRAM and off modes. Each mode had one cold request and three
warm repeats. Values are warm medians. All 96 output comparisons passed, with
maximum TVD 0.000143 and unchanged winners.

| Policy | Restore time | Complete native request |
|---|---:|---:|
| Host, 1024-token snapshot | 239.97 ms | 430.44 ms |
| VRAM, complete 1041-token snapshot | 1.87 ms | 149.74 ms |
| Off, repeated full prompts | 0.02 ms | 3022.44 ms |

Host and device snapshots represent about 184 MiB of tensor state. The device
path also caches the final 17 prefix tokens. Restoration is about 129 times
faster; complete warm requests are about 2.9 times faster. These are shared
context results, not the speedup of all independent requests.

The 4090 was observed at PCIe Gen 1 x4 during inference (92% GPU utilization),
with a Gen 1 target setting on its upstream port. The port supports Gen 4 x4.
The host boot configuration records prior 4090 PCIe failures. This hardware
finding helps explain the host-transfer cost; the tests do not isolate DDR4,
PCIe and per-tensor synchronization. PCIe settings were not changed. The 6000
was observed at Gen 5 x16 under load. Its final matched controls restored
host state in 19.22 ms and device state in 1.12 ms, a separate same-card result.
The complete warm requests took 163.51 ms and 113.28 ms respectively. All 96
comparisons passed, with maximum TVD 0.000054 and unchanged winners.

## Mappity and the secondary-cache decision

The actual application uses 693 frozen places, 425 judgments, 303 model
requests and 66,717 logical tokens per search. Its fine pass keeps each of 300
places in its own prompt. The game judges 60 candidates after a validity check.
The cache-only 6000 matrix used two runs per caller count and cache policy.
The final batching comparison used three runs each at four and eight callers,
in the order 4, 8, 8, 4, 4, 8.

| Final engine, 1 GiB device cache | Search median (range) | Game median (range) |
|---|---:|---:|
| RTX 4090, 4 callers | 26.23 s (26.15–26.36) | 4.20 s (4.20–4.28) |
| RTX 4090, 8 callers | 25.95 s (25.92–26.03) | 4.14 s (4.13–4.16) |
| RTX PRO 6000, 4 callers | 23.56 s (22.55–23.56) | 3.73 s (3.60–3.81) |
| RTX PRO 6000, 8 callers | 23.00 s (22.67–23.36) | 3.60 s (3.54–3.66) |

Mappity adopts eight callers, which were slightly faster on both cards. The
previous deployed 4090 engine at four callers measured 28.92 s/search and
4.48 s/game. Those are earlier measurements, not a simultaneous control.

On the cache-only 6000 engine, host mode averaged about 24.6 s/search; VRAM
budgets of 256 MiB, 1 GiB and 4 GiB averaged about 24.0–24.2 s. Off mode took
about 25.2 s. A 256 MiB cache held one state and had 11 evictions across four
application runs. A 1 GiB cache held all three reusable states (476 MiB charged),
with nine hits and no eviction after three initial misses. A 4 GiB budget did
not improve latency. The independent single-question fine pass bypasses this
exact-prefix cache and remains the main cost.

These measurements support a 1 GiB default and an explicit host option. They
do not justify an automatic RAM secondary tier for Mappity. More reusable
prefixes would need more than Prism's four ownership slots; a future extension
would need explicit snapshot handles and buffer release, or its own bounded
snapshot storage. Adding only RAM spill or a larger byte budget would not solve
Mappity's independent-prompt work.

The 4090 passed 16/16 explicit place-fact controls and 18/18 question-validity
controls in both host and VRAM modes. Game odds were finite and normalized.
The final 6000 check also passed all 16 place-fact and 18 validity controls.
Across the eight-caller 4090 VRAM runs, search score changes from the first
same-engine host run averaged 0.00217–0.00230 absolute, with maximum 0.01360.
Each had one near-0.5 crossing and retained nine or ten of the original top ten.
These unlabeled search results do not establish equal ranking quality or
validated calibration. See the [application checks](https://github.com/kortexa-ai/mappity/blob/main/experiments/README.md)
for the harness and full method.

## Reproduce the controls

Use the checked-out scripts with the built readout and pinned weights. Set
`CUDA_VISIBLE_DEVICES` to one full GPU UUID. Arrange sufficient free memory
before loading a model; these scripts do not manage services.

```bash
python scripts/benchmark_parallel.py --executable "$READOUT" --model "$MODEL" \
  --projector "$PROJECTOR" --parallel 4 --prefix-cache vram \
  --prefix-cache-mib 1024 --http --out "$OUTPUT/full"
python scripts/benchmark_cache.py --executable "$READOUT" --model "$MODEL" \
  --projector "$PROJECTOR" --prefix-cache vram \
  --prefix-cache-mib 1024 --out "$OUTPUT/churn"
```

Repeat the churn command with a 256 MiB budget and a distinct output directory.
Pass `--reference` to the parallel script to apply its frozen probability gates
against a previous result with the same fixture hash.
