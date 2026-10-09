# Independent sequence scaling on RTX PRO 6000

## Result

Raising the native sequence limit from four to 32 did not remove the Mappity fine-pass bottleneck. The measured work was 300 isolated place prompts and 51,643 input tokens. The original-order runs took 17–23 seconds. Grouping by prompt length reduced padding and call count; the best observed median was 16.51 seconds at four sequences.

CUDA activity captures show that GPU kernels occupied 94.3–95.5% of elapsed time. Quantized matrix multiplication and Gated DeltaNet account for most kernel time. Host/device copies took less than 0.09 seconds. Wider batches do not produce an order-of-magnitude improvement with this runtime and model.

These are native fine-pass measurements. They exclude HTTP, routing, the coarse pass and explanation selection. They are not new complete-search timings or a fresh Jev comparison. The production four-sequence limit and defaults remain unchanged.

## Method

- GPU: RTX PRO 6000 Blackwell, pinned by full UUID, beside the existing production services.
- Power: the existing 450 W ceiling remained fixed (the reported default is 600 W). No power, clock or fan settings changed. A captured status showed power capping active and no thermal slowdown. Temperatures and clocks varied under sustained load.
- Engine base: public `efdca285ded5770f990d7ee99b243cf7aaa359a0`, Prism `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`, the same model/projector and identity calibration as production.
- Native-only experimental limits: 4/8/16/32 sequences; one shared 16,384-token context; 1024/2048/4096/8192 token capacities. Python/API serving limits were not widened.
- Each prompt retains exactly one place. The fixture freezes the 300 places selected by the existing application. Every measured pass verifies all IDs and exactly 51,643 input tokens.
- One warm pass precedes three measured passes per unprofiled configuration. Length ordering uses token counts from that pass and adds a warm pass in the new order. Model load time is excluded.
- Each fresh worker runs a small canary, then 16 explicit positive/negative place-fact cases. At 32 sequences the controls repeat to exercise all slots.
- The sweep uses the listed order and repeats the four-sequence baseline at the end. It is not randomized. The repeat exposes timing drift; small differences must not be treated as a precise ranking.

Fixture SHA-256: `b92f000dd74889f0f8a2ac5aa8e1a184812426c1a9be3caee2fc672df9ac4550`.

## Sequence and token-capacity sweep

| Sequences | Token capacity | Median seconds (range) | Decode calls | Mean tokens/call |
|---:|---:|---:|---:|---:|
| 4 | 1024 | 17.39 (17.17–17.57) | 104 | 553 |
| 8 | 1024 | 18.21 (18.07–18.31) | 182 | 303 |
| 16 | 1024 | 19.50 (19.41–19.57) | 224 | 240 |
| 32 | 1024 | 20.92 (20.85–20.97) | 218 | 242 |
| 8 | 2048 | 18.81 (18.74–18.82) | 145 | 381 |
| 16 | 4096 | 19.98 (19.95–20.01) | 190 | 282 |
| 32 | 8192 | 21.60 (21.56–23.26) | 198 | 266 |
| 4 (repeat) | 1024 | 19.01 (18.96–19.01) | 104 | 553 |

The four-sequence median drifted from 17.39 to 19.01 seconds (9.3%). The data does not establish a meaningful four-versus-eight ranking under a stable clock. It does rule out a large throughput gain from simply raising the sequence limit in this workload.

With the unchanged 128-token total padding bound, wider groups of unequal-length prompts break into many small calls. Raising token capacity alone does not fill those calls.

## Length grouping and padding controls

All requests are still independent. Only their execution order changes. The third control raises the discarded-padding allowance from 128 to 1024 tokens per call; logits still come from each real prompt end.

| Sequences / capacity / padding bound | Median seconds (range) | Decode calls | Mean tokens/call | Total padding |
|---|---:|---:|---:|---:|
| 4 / 1024 / 128 | 16.51 (16.38–16.64) | 76 | 682 | 185 |
| 32 / 8192 / 128 | 18.16 (18.12–18.21) | 25 | 2106 | 1007 |
| 32 / 8192 / 1024 | 18.28 (18.22–18.31) | 10 | 5338 | 1733 |

The fullest 32-sequence control needs only ten decode calls, with 5,338 tokens per call, but still takes 18.28 seconds. Thus fragmentation has a cost, but removing it does not make the large model evaluate these tokens cheaply. Length grouping is a modest optimization candidate; this offline result is not a tested live scheduler or application change.

## CUDA activity attribution

Nsight Systems 2025.3.2 captures only one measured pass after warmup, with `--capture-range=cudaProfilerApi --cuda-graph-trace=node`. Profiled timings are separate from the unprofiled medians. Kernel and copy intervals were read from the exported SQLite activity tables. Union times avoid double-counting overlap.

| Captured activity | Four sequences, original order | 32 sequences, length order |
|---|---:|---:|
| Elapsed native pass | 19.150 s | 18.371 s |
| GPU kernel interval union | 18.057 s (94.3%) | 17.540 s (95.5%) |
| All GPU activity interval union | 18.265 s (95.4%) | 17.698 s (96.3%) |
| PQ2 matrix multiplication, summed kernel time | 12.143 s | 10.402 s |
| Gated DeltaNet, summed kernel time | 2.562 s | 3.104 s |
| Host-to-device plus device-to-host copies | 0.086 s | 0.088 s |
| Kernel instances | 295,544 | 67,135 |

The matrix kernels are `mul_mat_q<(ggml_type)142,...>`; the pinned runtime defines type 142 as PQ2_0. In the four-sequence capture they consume 66.8% of summed kernel time; Gated DeltaNet consumes 14.1%. The wider run reduces launch count more than fourfold without a corresponding elapsed-time reduction.

CUDA API duration is not additional CPU compute time: launch calls can wait on queue capacity, and stream synchronization waits on GPU work. Adding API time to kernel time would double-count elapsed work. The measured GPU interval coverage, rather than API duration alone, identifies the bottleneck.

The evidence points to PQ2 matrix throughput and the recurrent layers as the next performance targets. It does not prove why Jev is faster: Jev hardware, internal execution and model costs are not controlled here.

## Correctness and operating checks

- All 16 explicit fact cases passed in every configuration. All measured passes returned 300 finite probabilities and the same token count.
- Maximum probability change from the first four-sequence run was 0.015585; configurations produced zero to three crossings of 0.5. The repeated four-sequence baseline was identical. These are unlabeled search scores, so this is a numerical consistency check, not evidence of equal ranking quality.
- The existing 134 CPU tests passed before the trial. All experimental native builds completed with `-O2 -Wall -Wextra`.
- Minimum observed free VRAM was 29,043 MiB in the unprofiled controls and 29,045 MiB in the profiled controls. No OOM or new NVIDIA Xid occurred.
- Both process-owned GPU blocks ended with all nine recorded services healthy and every PID unchanged. No service needed to stop. The temporary workers exited and free VRAM returned to 51,130 MiB.

## Reproduce

The [fixture](fixture.json.gz), [experimental native patch](experimental.patch), [machine-readable summary](summary.json) and [benchmark](../../scripts/benchmark_sequence_scaling.py) preserve the experiment. The patch is a measurement-only extension; it is not enabled in the production worker or API. Apply it in an isolated checkout based on `efdca285`, using the same patched Prism runtime. It reproduces native source SHA-256 `2e79eede5a76455e44c497129c2ae3f6c8f859a244cc47f4334aaa3afbae8913`.

```bash
git apply results/sequence-scaling/experimental.patch
# Build this checkout with scripts/build.sh and set READOUT to its binary.
gzip -dc results/sequence-scaling/fixture.json.gz > /tmp/shingi-sequence-fixture.json
python scripts/benchmark_sequence_scaling.py --executable "$READOUT" \
  --model "$MODEL" --projector "$PROJECTOR" \
  --fixture /tmp/shingi-sequence-fixture.json --slots 4 --batch-tokens 1024 \
  --rounds 3 --out "$OUTPUT/s4"
python scripts/benchmark_sequence_scaling.py --executable "$READOUT" \
  --model "$MODEL" --projector "$PROJECTOR" \
  --fixture /tmp/shingi-sequence-fixture.json --slots 32 --batch-tokens 8192 \
  --length-order --reference "$OUTPUT/s4/results.json" --rounds 3 --out "$OUTPUT/s32"
```

Pin `CUDA_VISIBLE_DEVICES` to the full 6000 UUID. The benchmark requires at least 32 GiB free before loading and monitors a 10 GiB floor. Arrange the service/memory baseline outside this script. Use a distinct output directory per run. Set `SHINGI_TRIAL_PADDING_TOKENS=1024` for the wider-padding control. To capture activity, launch the Python benchmark under `nsys profile --sample=none --cpuctxsw=none --trace=cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop` and add `--profile --rounds 1` to the benchmark arguments.

Raw logs, every probability, GPU telemetry, Nsight reports, SQLite traces, input provenance and service receipts are retained under `~/Desktop/ai reports/shingi-sequence-scaling-2026-10-08/`. Initial sweep source: `619c0245966a33e6a283064bdcd61d1667d7700a`, executable `c371be579b6d5575629d0fba232cb605b6f1240ff41699dd195be4428e18006f`. Follow-up source: `677fcb30996d9d879c5524d24d857b550a2d009a`, executable `a7d048fe913c116ad6be578c3f4c152086a73b089bf756fca82f154d5252a7c2`.
