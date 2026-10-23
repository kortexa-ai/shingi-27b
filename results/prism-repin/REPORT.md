# Prism runtime repin: 4fbda1292 against d8f26eec on the RTX PRO 6000 and the RTX 4090

Owner: https://github.com/kortexa-ai/shingi-27b/issues/7. Measured on smarty: RTX PRO 6000 2026-10-23 03:56 to
04:00 UTC, RTX 4090 05:30 to 05:35 UTC in an authorized production block.

The owned Prism runtime moves from `d8f26eec76da6d09bb708bcba51ef64b8cd868a3` (2026-09-09)
to `4fbda1292562cc1f23904d5867fb79603737547a` (2026-10-07), the revision the public
Bonsai 2 9B CUDA harness vendors. The quantized device-state correction stays. The SM120
recurrent-column patch is retired because the new revision derives the four-column Gated
DeltaNet layout for every NVIDIA Ampere or newer GPU (upstream #216). `readout.cpp` is
unchanged; both build stamps carry the same source hash.

## Method

- Isolated runtime home and isolated clone at branch commit `1464055`; nothing in the
  production engine directory was modified. The baseline arm links the production
  `runtime-gdn-20261009` libraries (d8f26eec, architectures 89;120). The candidate arm links
  the new build (4fbda1292, architectures 89;120).
- Both arms run the readout with `results/sequence-scaling/experimental.patch` applied,
  which adds timing fields and does not change the decode path.
- Fixture `results/sequence-scaling/fixture.json.gz` (SHA-256 `b92f000d…`): 300 independent
  Mappity fine prompts, 51,643 input tokens, 16 explicit fact controls. Four sequences,
  1024-token batches, 16,384-token context, 1 GiB VRAM prefix cache.
- Order: baseline, candidate, baseline, candidate. Each run loads the model, passes the
  controls, runs one warm pass, then two measured rounds.
- GPU `GPU-a71210ca-e14a-755a-88bb-77f53a2102f6` with its existing 450 W cap. No service was
  stopped. The RTX 4090 production worker was not touched.

## Results

| Run | Runtime | Round 1 | Round 2 | p50 / p95 exchange | Controls | Max probability difference to baseline-1 |
|---|---|---:|---:|---:|---:|---:|
| baseline-1 | d8f26eec | 16.774 s | 18.191 s | 238 / 313 ms | 16/16 | 0 |
| candidate-1 | 4fbda129 | 17.187 s | 17.320 s | 237 / 265 ms | 16/16 | 0.00299 |
| baseline-2 | d8f26eec | 17.670 s | 17.761 s | 244 / 272 ms | 16/16 | 0 |
| candidate-2 | 4fbda129 | 17.725 s | 17.798 s | 244 / 272 ms | 16/16 | 0.00299 |

Every run decoded the same 51,643 tokens in 104 decode calls with 5,893 padding tokens.

Speed is unchanged within measurement noise. The two baseline runs differ from each other by
more than either arm differs from the other. The repin is not a speedup and not a regression
on this GPU and shape.

Numerics are not bit-identical. The two baseline runs reproduce each other exactly. The
candidate changes every one of the 300 probabilities: mean absolute difference 0.00029,
maximum 0.00299, 11 prompts above 0.001, zero crossings of 0.5. Six prompts sit within 0.01
of 0.5 and none of them crossed. The difference is within the 0.02 drift gate used by the
earlier runtime experiments. These prompts are unlabeled; this is not a task-quality result.

Upstream changes between the two pins that touch this workload: flash attention reads the
Q8_0 K/V cache in place instead of converting it to F16 per batch (#221), the recurrent state
gather is folded into the Gated DeltaNet kernel (#220), and the four-column recurrent layout
applies to all Ampere or newer GPUs (#216; SM120 already had it through the retired patch).

## Health

Free memory on the 6000 was 51,032 MiB before and after; the lowest sampled value during a
run was 40,585 MiB. Temperature rose from 36 to 78 °C under the 450 W cap. `ktxsvc list`
was identical before and after. The 4090 kept 18,157 MiB in use throughout.

## RTX 4090

Franci authorized production downtime on the RTX 4090 for this measurement. The block stopped
`legolm/legolm.miso`, `models/shingi-27b`, `tts.server` and `asr/qwen` at 05:30 UTC, ran the same
four-run A/B pinned to `GPU-afb49bc6-cd89-6584-99cc-a0f03592a010` with the engine's 14/4 GiB
gates, and restored the services after the last run. The arms are the same two binaries as on the
6000; the clone was at branch commit `2d5dcff`, where the benchmark takes the engine's
GPU-size-aware gates instead of the fixed 32 GiB check.

| Run | Runtime | Round 1 | Round 2 | p50 / p95 exchange | Controls | Max probability difference to baseline-1 |
|---|---|---:|---:|---:|---:|---:|
| baseline-1 | d8f26eec | 21.392 s | 21.382 s | 288 / 322 ms | 16/16 | 0 |
| candidate-1 | 4fbda129 | 20.287 s | 20.279 s | 275 / 307 ms | 16/16 | 0.00269 |
| baseline-2 | d8f26eec | 21.377 s | 21.394 s | 288 / 322 ms | 16/16 | 0 |
| candidate-2 | 4fbda129 | 20.290 s | 20.309 s | 274 / 307 ms | 16/16 | 0.00269 |

Same 51,643 tokens, 104 decode calls and 5,893 padding tokens in every round. The repin is 5.1 %
faster on wall time (21.39 s to 20.29 s per round) and 5.3 % on decode time (20.72 s to 19.62 s);
run-to-run noise is below 0.1 %. This is the four-column recurrent layout that upstream #216
brings to Ada; production ran one column wide on this GPU until now.

Numerics: the two baseline runs are bit-identical to each other. The candidate changes every
probability: mean absolute difference 0.00032, maximum 0.00269, 5 prompts above 0.001, zero
crossings of 0.5.

Health: 22,569 MiB free after the stops, 12,430 MiB free with the model loaded, lowest sampled
12,326 MiB, 22,569 MiB free again after the runs. No Xid event on PCI 0000:06:00. The 6000 and its
services were not touched.

Restore: the block restarted all four services at once and `models/shingi-27b` failed its preload
gate (11,536 MiB required, 11,044 MiB free once TTS, ASR and Miso had loaded); systemd stopped
retrying after three attempts. It was restored by hand in boot order (stop `tts.server`, start
Shingi, wait for health, start `tts.server`). All four services answered 200 at 05:41 UTC and the
card returned to its 21,675 MiB baseline. The block script now restores one service at a time in
stop order.

## Limits

Two GPUs, one shape, two rounds per run. No hardware-counter profile.

## Reproduction

```sh
export CUDA_VISIBLE_DEVICES=GPU-FULL-UUID
SHINGI_HOME=/path/to/isolated/home SHINGI_CUDA_ARCHITECTURES='89;120' bash scripts/build.sh
git apply results/sequence-scaling/experimental.patch   # private checkout only
c++ -std=c++17 -O2 src/native/readout.cpp -I$PRISM/include -I$PRISM/ggml/include \
  -I$PRISM/vendor -I$PRISM/tools/mtmd -L$PRISM/build/bin -Wl,-rpath,$PRISM/build/bin \
  -lmtmd -lllama -lggml -lggml-base -o readout-trial
uv run python scripts/benchmark_sequence_scaling.py --executable readout-trial \
  --model shingi-27b.gguf --projector mmproj.gguf --fixture fixture.json \
  --out results-dir --slots 4 --batch-tokens 1024 --rounds 2 [--reference baseline/results.json]
```

[summary.json](summary.json) holds every round, the control counts, memory samples and GPU
state. Raw results, logs and checksums are retained in the operator's
`shingi-cudafast-audit-2026-10-22/prism-repin-ab` report directory.
