# CUDA kernel investigation on RTX PRO 6000

Measured on 2026-10-09 with the same 300 isolated Mappity place prompts and 51,643 input tokens as the [sequence experiment](../sequence-scaling/REPORT.md). These are native fine-pass times, not full HTTP search times.

The useful change is in Gated DeltaNet: reuse each loaded query, key and gate across four state columns. Prism already uses this implementation on DGX Spark. Enabling it on compiled SM120 kernels gives identical answers and reduces the sampled recurrent kernel time by about 27%. Warm full-pass measurements improve from about 19.27 to 18.07 seconds, about 6%. It does not close the large gap to the older Jev measurement.

The build now applies this change to Shingi's owned Prism runtime. The host grid uses the compiled architecture to match device column ownership, including builds which run older CUDA code on a newer GPU. The SM89/RTX 4090 kernel keeps its existing arithmetic. The smaller PQ2 tiles and cuBLAS routing below are rejected experiments, not production defaults.

## What makes the kernels slow

The earlier whole-pass trace put roughly 12.14 seconds in PQ2 matrix multiplication and 2.56 seconds in Gated DeltaNet. GPU kernels covered 94.3% of wall time. The new Nsight Compute samples explain two distinct costs:

| First sampled kernel | PQ2 matrix multiply | Gated DeltaNet | Four-column Gated DeltaNet |
|---|---:|---:|---:|
| Device time | 193.95 us | 387.49 us | 282.72 us |
| Registers per thread | 255 | 40 | 56 |
| Achieved occupancy | 16.66% | 88.40% | 58.52% |
| SM clock during sample | see raw counters | 2.513 GHz | 2.571 GHz |
| Load/store instruction pipeline | see raw counters | 93.06% | 77.99% |

PQ2 does use integer Tensor Cores. It expands packed weights, loads block scales, and applies scales to short integer dot products. The main 128-column kernel uses 255 registers per thread and one block per SM. The sample reports 39.73% Tensor Core activity during active cycles and about 6.23 MB of local load/store sector traffic. Nsight identifies that local traffic as register spilling. These spills stay in the GPU memory hierarchy; they are not CPU/RAM cache transfers. DRAM bandwidth in this sample is only 12.27 GB/s. This is not evidence of saturated off-card bandwidth.

Gated DeltaNet has a different limit. The original kernel assigns one state column to each warp. Many warps therefore reload the same query, key and gates for each token. The load/store instruction pipeline is nearly full, although DRAM traffic is only 24.46 GB/s in the sample. Four-column reuse reduces that repeated work. Occupancy falls, but the kernel gets faster. Higher occupancy alone is not the objective.

The recurrent time reduction remains about 25% after normalizing these two samples by their measured SM clocks. GPU clocks and caches were not locked. Do not treat one profiled launch as a whole-model speed estimate.

## Controlled alternatives

Each entry is the median of two measured passes after warmup. All original-order cases use four sequences and a 1024-token batch. All produce 300 probabilities and process the same logical input tokens.

| Runtime | Fine pass | Result |
|---|---:|---|
| Original, first control | 17.810 s | Cold operating point |
| Original, later controls | 18.779 / 19.265 / 19.271 s | Sustained-load drift |
| Four-column recurrent kernel | 17.455 / 18.070 s | All 300 probabilities bit-identical |
| PQ2 64-row tiles, maximum width 32 | 25.066 s | Rejected: slower |
| PQ2 64-row tiles, maximum width 64 | 21.819 s | Rejected: slower |
| PQ2 64-row tiles, maximum width 128 | 20.587 s | Rejected: slower |
| cuBLAS, FP16 inputs and FP32 accumulation | 28.020 s | Rejected: slower |

The smaller-tile trial changes both tile geometry and work partitioning: 128 threads, 64 output rows, and direct output tiling instead of stream-k. It is not a pure register-count experiment. It increases tile work and changes accumulation order. Its maximum probability difference is 0.01182, with one crossing of 0.5. The cuBLAS trial expands weights into GPU scratch memory for each operation. Its maximum difference is 0.01035, also with one crossing. These prompts are not labeled task-quality data.

To avoid rejecting cuBLAS merely for small batches, a second comparison uses 32 sequences, an 8192-token batch, length grouping and a 1024-token padding allowance. Both runtimes use only ten decode calls, averaging 5338 tokens per call:

| Wide configuration | Fine pass |
|---|---:|
| Original PQ2 kernel | 18.344 s |
| cuBLAS | 20.840 s |

Larger batches reduce the cuBLAS penalty but do not reverse it. Both conversion work and the different matrix kernel are included in this comparison. This experiment does not establish that every possible cuBLAS strategy, prepacked representation or tile configuration is slower.

## Validation and limits

- Existing CPU-reference operator tests pass: 39 supported Gated DeltaNet cases and 45 PQ2 matrix cases. The cuBLAS alternative also passes the 45 matrix cases. Unsupported operator layouts are not counted as tested.
- Every full-workload run passes the explicit fact controls. The shipped recurrent change preserves every probability in all six measured 300-prompt passes exactly, including two passes against the final multi-architecture release build. That build takes 16.95 and 17.18 seconds after the GPU cools during compilation; these are not comparisons with the hot controls.
- Shingi's 134 CPU tests pass, including managed-patch upgrades and rejection of unrelated source edits before any file is changed.
- The experiments keep at least 26.0 GiB free on the 6000. No new NVIDIA Xid or OOM occurs. All nine monitored services stay healthy with unchanged PIDs. No production service is stopped for these measurements.
- The 6000 keeps its existing 450 W limit. The original control drifts from 17.81 to 19.27 seconds under sustained load. The approximately 6% full-pass gain is a warm comparison, not a universal speed guarantee.
- The tests use isolated place prompts. They do not change prompt sharing, model weights, calibration, production sequence limits or production batch size.

## Reproduction

Use Prism revision `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`, CUDA 13.0.88 and the pinned Shingi weights. The fixture SHA256 is `b92f000dd74889f0f8a2ac5aa8e1a184812426c1a9be3caee2fc672df9ac4550`. Obtain it from `results/sequence-scaling/fixture.json.gz`.

Apply the [sequence instrumentation patch](../sequence-scaling/experimental.patch) only in a private Shingi checkout and compile its readout. Use separate runtime library directories for each candidate; never replace libraries used by a running service. The benchmark wrapper selects a runtime with `--library-path`, a smaller-tile width with `--mmq-j`, and experimental cuBLAS routing with `--cublas`.

The three patches in this directory describe the experiment changes against pristine pinned Prism source. Apply the existing quantized device-state correction as well. The GDN experiment used the numeric SM120 constant; the managed production patch uses the named constant and matches the host grid to the highest compiled architecture. For the original control, use the quantized device-state correction without the GDN change.

For example, with a UUID-pinned GPU and externally enforced memory/service checks:

```sh
python scripts/benchmark_kernels.py \
  --library-path /path/to/isolated/runtime/build/bin \
  --executable /path/to/instrumented/readout \
  --model /path/to/shingi-27b.gguf --projector /path/to/mmproj.gguf \
  --fixture /path/to/fixture.json --out /path/to/new/results \
  --slots 4 --batch-tokens 1024 --rounds 2
```

Nsight Compute 2025.3.1 captures only selected launches after warmup, using `--profile-from-start off`, `--set full`, `--clock-control none`, and `--cache-control none`. Profile the Python harness and its child readout so profiler output cannot enter the native JSON pipe. Hardware-counter runs are separate from latency runs. See NVIDIA's [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) for replay, clock and cache effects.

[summary.json](summary.json) contains every latency trial and health result. [ncu-summary.json](ncu-summary.json) contains selected counters with their original units. Full reports, logs and checksums are retained in the operator's `shingi-kernels-2026-10-09` report directory.
