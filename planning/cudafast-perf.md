# cudafast kernel audit for Shingi (2026-10-22)

Owner: https://github.com/kortexa-ai/shingi-27b/issues/7

Read-only review of the promoted submissions on the public Bonsai 2 9B CUDA
leaderboard, https://www.yukon.org/cudafast?benchmark=bonsai2-9b&platform=cuda,
against Shingi's prefill workload. No kernel was run. Nothing here is a measured
Shingi result; every number comes from the harness source, upstream Prism pull
requests, or Shingi's own earlier reports.

## What the benchmark measures

- Harness: `Layr-Labs/cudafast-bonsai2-9b-engine`, main at `759629b` on 2026-10-21.
  Vendored PrismML llama.cpp at `4fbda1292` (2026-10-07). Shingi pins `d8f26eec` (2026-09-09).
- Target: Ternary-Bonsai-2-9B PQ2_0 with MTP head, DGX Spark GB10 (sm_121), batch 1,
  f16 KV, context 8192, n_batch 2048, n_ubatch 512, speculative decoding (DFlash2 depth 15 at the top).
- Score: `prefill_gain^0.25 * decode_gain^0.75`. Decode dominates.
- Leaderboard headline: 421.5%. Decode 85.9 -> 576.7 tok/s, prefill 6,852.6 tok/s, 10.3 tokens
  per draft round. Prefill across the shown entries rose from 3,098 to 6,853 tok/s.
- 35 promoted submissions since the 2026-10-17 baseline reset (`ca07598`). Each lands as one
  "Validate/Accept submission <uuid>" commit on main. Commit bodies carry no measurements.
  Per-technique numbers live only in Yukon submission notes (CLI installed, not logged in).

## Shingi workload and profile (for fit)

- Ternary-Bonsai-2-27B PQ2_0: hidden 5120, FFN 17408, 64 layers, attention 24/4 heads x 256,
  GDN 16 key heads x 128 and 48 value heads x 128, conv 4, 1-in-4 full attention.
- Prefill only: 1024-token ubatches, up to 4 sequences, kv_unified, Q8_0 K/V, flash attention,
  final-token logits only. RTX 4090 (sm_89) production, RTX PRO 6000 (sm_120) experiments.
- Profile (6000, 4 seqs, 300 prompts): kernels 94% of wall; PQ2 MMQ 67% (128-column tile,
  255 regs/thread, 1 CTA/SM, spills), Gated DeltaNet 14%, copies 0.5%.

## Global finding

Every new CUDA kernel in the submissions is host-gated `cc == GGML_CUDA_CC_DGX_SPARK` (1210).
sm_89 (890) and sm_120 (1200) never pass. Nothing transfers by rebuilding; everything needs
a de-gated fork. Several kernels also hardcode the 9B shapes (hidden 4096, 16/4 heads,
32 GDN heads) and single-sequence operation.

## Ranked candidates for Shingi

### 1. PQ2_0 J=128 MMQ rewrite, `mmq-pq2-j128.cuh` (bit-exact claim)

- Replaces `mul_mat_q<PQ2_0,128,false>`: CTA 64x128, 8 warps, `__launch_bounds__(256,2)`,
  <=128 regs, 2 CTAs/SM (16 warps/SM), shared int8 weight tile, cp.async or TMA activation feed,
  same stream-k K partition, reuses the unchanged stream-k fixup. Epilogue uses the 1.5*2^23
  trick to reproduce `sum += C*dA*dB` bit for bit. This is the exact shape of our bottleneck.
- Gate: `mmq.cuh:1784` `cc == GGML_CUDA_CC_DGX_SPARK`. Device code is Ampere+ except TMA
  (`cp.async.bulk`, sm_90+ PTX) in the stream-k variant, and L2 hints / PDL under sm_121 ifdefs.
- Shapes: needs `nrows % 128 == 0`, `K % 256 == 0`; 27B weights pass (5120, 17408, 6144, 1024).
- Variant selection at our shapes (1024 tokens, 8 column tiles, old 128-row tile count):
  FFN gate/up (17408 rows) -> pure tiling on both GPUs (94% / 96% wave efficiency).
  ffn_down (5120 rows) -> stream-k on both (83% / 85%). attn q (6144) -> pure on the 4090, stream-k on the 6000.
  So on the 4090 the two largest GEMMs per layer already take the cp.async path; the rest needs
  a cp.async port of the stream-k feed. The 6000 should run both variants (verify TMA on sm_120).
- Shared memory: 45 KB (pure) / 49 KB (stream-k) per CTA. Two CTAs sit at the 100 KB per-SM
  limit on sm_89 and sm_120. Confirm 2 CTAs/SM with the occupancy calculator.
- Options: `GGML_CUDA_PREFILL_COMPACT` (lower register pressure, exact). Skip D1 and TMAX
  (D1 changes activation scale granularity, not exact).
- Commits: 3aa563e (10-17), 83d799c, a856990, 8539fa3 (10-18), 2c62b6c (10-20), 0870181 (10-21).

### 2. Gated DeltaNet

- T1 batched warp reductions + hoisted v loads in the generic serial kernel. Ungated, exact.
  Commit 6690be4. Small ILP win.
- Upstream Prism #216 (2026-09-23): 4-column warp layout on all Ampere+, +6% prefill on Ada
  for the 27B in upstream's measurement. Our production patch enables it only on sm_120; the
  4090 still runs the 1-column path.
- T2 "L8" lane-permuted serial kernel: 4 lane groups of 8, 6 shuffles per token instead of 40,
  cp.async token staging. Exact (same reduction tree). Multi-sequence and K==1 capable as written.
  Gates: `cc == DGX_SPARK`, `keep_rs` (speculative contexts only), S_v 128, n_tokens >= 17.
  Our Nsight profile is LSU-pipe bound, which includes shuffles, so this fits.
  Commits 6690be4, 923d628, 83d799c, 8539fa3, 239e6e8.
- T4 "N1" chunked WY-form f32 prefill scan: 32-token chunks, parallel inside a chunk, serial
  across chunks per head. The only change that removes the per-token serial dependency.
  Not bit-identical (test tolerance atol 2e-5, rtol 2e-4). Gated to sm_121, one sequence,
  keep_rs, T >= 64, K < T. Port needs a sequence grid dimension and a K==1 launcher, plus
  our own numeric validation on decision outputs. Commit 239e6e8.
- T10 conv-front fusion (concat + snapshot copies + ssm_conv + silu + q/k l2norm in one kernel):
  exact, prefill path for T >= 64, gated sm_121 and one sequence. Small.
- Skip: half prefill (fp16/TF32 GDN, "intentionally numerical", 9B head counts hardcoded),
  lazy rollback, dedup, L8V, G1 (all speculative verify-step work).

### 3. Last-attention-layer query narrowing (`llama-last-attn.h`)

- At the last layer (full attention for the 27B too), run Q projection, MRoPE, flash attention,
  sigmoid gate and `wo` only for the output row(s); K/V projections and cache writes still run
  on every row. Baseline already narrowed the last layer's FFN.
- Host side, no arch gate. Hardcoded: `n_seqs == 1`, `n_outputs == 1`, hidden 4096, 16/4 heads,
  head 256, PQ2_0 wq/wo, no LoRA. Not bit-exact (2e-4 relative on attention output).
- For us: generalize to up to 4 output rows and 27B hparams. Saves one layer's attention over
  1024 queries per ubatch. Modest, cheap. Commit 8539fa3.

### 4. Fused gate/up GEMM with SwiGLU epilogue + residual ADD fold

- Ride on candidate 1. Exact in F32 mode. Requires the gate/up GEMM to take pure tiling (true
  for the 27B at 1024 tokens). Costs a row-interleaved copy of ffn_gate|ffn_up: about 3 GB extra
  VRAM for the 27B. Fine on the 6000. Not on the production 4090 (about 2.4 GB free after load).
- Commits a856990 (GLU), 83d799c (ADD fold).

### 5. Small exact, arch-neutral pickups

- RMS_NORM*w + MROPE fusion (norm+rope part only; the K/V store part needs f16 or f32 cache).
- ADD+RMS_NORM+MUL in-place alias relaxation (one residual read saved per layer).
- CUDA-graph cache key includes node count and shape (prevents prompt/tail graph eviction).
- `token_embd.weight` on the GPU via buffer-type override (removes a CPU split per ubatch).
- Ranged KV clear instead of whole-buffer memset between requests (KV part is arch-neutral).
- Boot warm-up pass before the socket opens (service pattern).
- Expected combined effect: small. Copies and glue are under 5% of our profile.

## Not applicable to Shingi

- All speculative decoding work: DFlash2 device walk/rescore, MTP, lazy rollback, dedup,
  decode-layout J16 kernels, PTQ1 re-encoding, QKV joint decode copy, decode gate side-streams,
  fattn-lowstage (16 queries, f16 KV), half-precision FA partials (16 queries, f16 KV), top-k select.
- FWHT + Q8 quantize fusions: byte-exact but hardcode N=4096 butterflies. The 27B hidden size is
  5120, a Kronecker Hadamard size. Generalization is real work for a second-order gain.
- D1 activation quantization: changes numerics.
- PDL extensions: no effect on sm_89 (Hopper floor); marginal for 1024-token prefill on sm_120.
- Nothing in the submissions touches the Q8_0-KV attention path we use; that is baseline code.

## Upstream Prism changes since our pin worth taking in the same fork

- #216 (09-23) 4-column GDN layout on all Ampere+ (Ada path for the 4090).
- #221 (09-29) flash attention reads q4_0/q8_0 K/V in place; our pin converts the quantized
  cache to an F16 scratch per ubatch during prefill. Also the GDN state gather fusion (#220).
- #318 (10-05) GDN gather fix for multi-image hybrid VLM prompts (only needed with #220).
- #306 (10-09) PQ2_0 mat-vec for 3-8 columns on Ada: tail batches only, minor.
- #330 (10-19) opt-in f16 prefill FA for quantized KV: measured on Turing only.

## Suggested route

1. Re-pin Prism to `4fbda1292` (the harness baseline), rebuild, and re-run the existing
   kernel benchmark and engine tests. This alone brings #216 and #221. Issue #7 does this step.
2. Layer candidate 1 with the host gate widened to sm_89/sm_120 (pure variant first on the 4090,
   both on the 6000). Measure with `scripts/benchmark_kernels.py` and the 300-prompt pass.
3. Layer T1, T2 and the Ada 4-column GDN path. Then decide on N1 after a numeric comparison.
4. Last-attn narrowing generalized to 4 outputs.
5. GLU pair fusion on the 6000 only.

## Evidence

- Harness clone used for diffs: `https://github.com/Layr-Labs/cudafast-bonsai2-9b-engine`
  main `759629b`; baseline `ca07598`; per-submission commits listed in the appendices.
- Prism upstream: `https://github.com/PrismML-Eng/llama.cpp` branch `prism`, PRs #216, #221,
  #306, #307, #318, #330, #364, #369.
- Qwen3.5-27B config: `https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json`.
- Shingi profile: `results/kernel-investigation/REPORT.md`,
  `results/sequence-scaling/REPORT.md`.

---

# Appendix A: PQ2 MMQ and activation quantization audit

Repo root: the harness clone; all paths relative to `llama.cpp/ggml/src/ggml-cuda/` unless noted.

## 0. Baseline facts

- The 128-column PQ2_0 MMQ kernel our profile measured is `mul_mat_q<PQ2_0, 128, false>` configured by `mmq-config-ampere.cuh:51`: `CASE(PQ2_0, nthreads=256, occupancy=1, I=128, J=128, SRAM_Q8_0, K_vram=256, stream_k=true)`. Blackwell (sm_120) and GB10 (sm_121) tables have no PQ2_0 rows and fall through to this same Ampere row (`mmq-config-blackwell.cuh:36,39-41`). No submission touched any config table, so on sm_89/sm_120 the generic kernel still runs at 8 warps/CTA, 1 CTA/SM.
- `cc` is `100*major+10*minor`: sm_89 = 890 (`GGML_CUDA_CC_ADA_LOVELACE`), sm_120 = 1200 (`GGML_CUDA_CC_BLACKWELL`), sm_121 = 1210 (`GGML_CUDA_CC_DGX_SPARK`, `common.cuh:61`). Every new matmul route is gated `cc == GGML_CUDA_CC_DGX_SPARK` on the host; nothing uses `>=`.
- `ggml_cuda_should_use_mmq` and `should_use_mmvq` are unchanged; PQ2_0 goes MMVQ for ne11 <= 8, MMQ above. 1024-token ubatches select J=128 as before.
- PDL device primitives compile only for `__CUDA_ARCH__ >= 900` (`common.cuh:124-134`): no-ops on sm_89, active on sm_120.

## 1. `mul_mat_q_pq2_j128`: rewritten J=128 PQ2_0 GEMM

- File `mmq-pq2-j128.cuh` (new, 1102 lines). Dispatch `mmq.cuh:1782-1795` inside `launch_mul_mat_q<PQ2_0,128,false>`.
- Mechanism (header `:3-27`): CTA = 64 rows x 128 columns, 8 warps, `__launch_bounds__(256, 2)` (`:719`), <=128 regs/thread, 2 CTAs/SM, 46-50 KB smem per CTA. Each warp owns a 32x32 output block. Weights unpacked once per CTA into a shared int8 tile (same 2-bit->int8 `__byte_perm` codec as `ggml_cuda_mmq_load_tiles_pq2_0`, `:174-195`) from raw words fetched one K block ahead (`:154-172`); activation tile double-buffered with `cp.async` (pure tiling) or one TMA bulk copy per K block on an mbarrier (stream-k) plus a transposed copy of the D4 scales (`:541-611`). mma = `m16n8k32 s8.s8.s32` via `ldmatrix`. Work partition: grid = 2x the old grid, CTA (b,q) runs the old block b's stream-k segments for row half q (`:733-751`), writes `tmp_fixup[b]` for trailing segments and reuses the unchanged `mul_mat_q_stream_k_fixup<type,128,false>` (`:1045-1053`). Host grid rule identical to the old one (`:971-978`, 90% wave-efficiency test).
- Epilogue: accumulator seeded with 0x4B400000 (1.5*2^23) so the int32 result is the float `12582912 + idot` exactly for |idot| <= 2^22; then `fma(C, dA, -12582912*dA)` = RN(idot*dA) and a second FFMA by dB (`:16-21`, `:276-281`). Claims bit-identity with the old `sum += C*dA*dB` including signed zero.
- Gates: `mmq.cuh:1784` `cc == GGML_CUDA_CC_DGX_SPARK && ggml_cuda_mmq_pq2_j128_use(args, nsm)`; `_supported` (`:959-968`): PQ2_0, no ids/channels/samples, `nrows_x % 128 == 0`, `ncols_x % 256 == 0`, y 16-byte aligned, sizes < 2^31. Device-side the only sm_121 blocks are L2 hints + PDL sync (`:753-777`); mma/ldmatrix/cp.async paths are Ampere/Turing. TMA (`cp.async.bulk`, `:96-112`) needs sm_90+ PTX.
- Exactness: claimed bit-identical (harness test `prefill_compact_equivalence.cu` compares bitwise).
- Commits: 3aa563e (cp.async only), 83d799c (TMA stream-k + ADD fold), a856990 (GLU epilogue), 8539fa3 (COMPACT), 2c62b6c (D1), 0870181 (TMAX).
- Applicability: applicable after removing the sm_121 host gate; on sm_89 the stream-k variant needs a cp.async replacement for the TMA Y feed. No measured numbers in code.

## 2. COMPACT compute loop

- `ggml_cuda_pq2_ws_compute_compact` (`:287-346`): same math with `#pragma unroll 1` over the four 32-value sub-blocks for lower register pressure. `GGML_CUDA_PREFILL_COMPACT` default 1. Bit-identical. Commits 8539fa3, 91eb311.

## 3. Residual ADD folded into the J=128 store

- `mmq_args.add_res/add_dst` (`mmq.cuh:1615-1619`), epilogue `ggml_cuda_pq2_j128_store_add` (`:636-662`), stream-k variant `mul_mat_q_pq2_j128_fixup_add` (`:853-949`). Matcher `ggml_cuda_mmq_add_fusable` (`ggml-cuda.cu:4385`): `cc == DGX_SPARK`, `mm->ne[1] > 16`, PQ2_0 weight, MMQ route, registered D4 activation. `GGML_CUDA_MMQ_ADD_FUSION=0` disables. Exact (one rounded add). Commit 83d799c.

## 4. Prefill gate/up pair fused GEMM with SwiGLU epilogue

- Load time: `ggml_cuda_glu_pair_register` (`mmq.cu:1029`) builds a row-interleaved copy of ffn_gate|ffn_up; registered from `llama-model.cpp:2256-2285` (`LLAMA_PREFILL_GLU_PAIR`), gated `cc == DGX_SPARK`, PQ2_0, `nrows % 128 == 0`. Doubles FFN gate/up weight memory.
- Run time: `ggml_cuda_try_prefill_glu` (`ggml-cuda.cu:4005`) matches MUL_MAT,MUL_MAT,GLU(SWIGLU) on one registered D4 activation, requires pure tiling (`mmq.cu:1010-1016`), launches `mul_mat_q_pq2_j128<PURE,ADD=false,GLU=true>` over 2F rows; epilogue `store_glu` (`:675-708`) writes `silu(gate)*up`. Optional F16 storage (`GGML_CUDA_PREFILL_GLU_F16`) is not bit-exact.
- Commits a856990, 0870181. Applicable after removing the gate, exact in F32 mode.

## 5. Prefill "plane" weight copy + TMA weight feed (`mmq-pq2-plane.cuh`, TMAX)

- Load time re-layout of PQ2_0 weights (lossless) for the stacked gate/up and, since 0870181, for attn_qkv/attn_gate/ssm_out/ffn_down/attn_q/k/v/attn_output (`llama-model.cpp:2289-2316`, `LLAMA_PREFILL_PLANE`); another full weight copy in VRAM. Run time `mul_mat_q_pq2_plane` (`:110-186`): Y tile and code tile arrive by TMA. TMAX for non-GLU only with D1. Needs TMA (sm_90+): plausible on sm_120 after de-gating, not on sm_89. Commits 8539fa3, 91eb311, 0870181.

## 6. D1 activation quantization (not exact)

- `ggml_cuda_prefill_d1_enabled(rows)` (`quantize.cu:1047-1053`): one FP32 scale per 128 values instead of per 32; logs `numerical_change=true`. Commit 2c62b6c. Not recommended without evaluation.

## 7. Fused FWHT (Hadamard) + Q8_1-D4 quantization producers

- `ggml_cuda_try_fwht_mmq` (`ggml-cuda.cu:3578`, `GGML_CUDA_FWHT_MMQ_FUSION`, hard `cc != DGX_SPARK -> return 0`) replaces the F32 transform + N separate `quantize_mmq_q8_1` launches with one kernel that writes the `block_q8_1_mmq` D4 buffer once, shared by all consumers. Kernels in `quantize.cu`: `fwht_quantize_mmq_d4` (`:1102`), `_rms` (`:1218`, >16 rows), `_add_rms` (`:1324`, <=16 rows), `_gnorm` (`:1262`), `_glu_perm` (`:1160`), `_sigmoid_gate` (`:1494`). All hardcode N = 4096 and the 1/64 transform scale. Bit-identical to standalone FWHT -> quantize per `fwht_mmq_equivalence.cu`. Butterfly variants in `fwht.cuh` (ffe5e04, 2c62b6c) are shared-memory reorganizations.
- Commits f685132, 923d628, 83d799c, ffe5e04, 2c62b6c. Applicable only after generalizing N (27B is 5120).

## 8. Plain Q8_0 activation reuse

- Q8_0 weights, `1 < ne11 <= 32` only. Irrelevant. Commits ecc1b50, 91eb311.

## 9. Generic `mul_mat_q` changes

- L2 prefetch hints (`mmq.cuh:954-983`, `:1083-1102`), funnel-shift tile loads (`mmq-load-tiles.cuh:206-304`), PDL launches (`mmq.cuh:1682-1694`), stream-k fixup partition table (`GRID == 48`, decode). All under sm_121 ifdefs or `cc == DGX_SPARK`. Register count, occupancy, tile geometry and stream-k scheduling of the generic J=128 kernel are untouched on sm_89/sm_120.

## 10. GDN gate projections scheduled early

- `llm_schedule_prefill_gates` (`llama-prefill-gates.h:10-32`), from `qwen35.cpp:497`. Gates `n_seqs == 1`, `n_tokens > 16`, `ne[0] == 4096`, weights `[4096,32]`. Exact reorder. Commit 8539fa3.

## 11. Decode-only (all `cc == DGX_SPARK`)

- `mul_mat_q_pq2_j16`, decode-layout weight copies + PTQ1_0 re-encoding + ring J=16 kernels (`mmq-j16-dl.cuh`, `mmq-ptq1-j16.cuh`), Q8 TM64 ring, fixup-pair, decode SwiGLU ring, stacked gate|up decode launch, QG|K|V grouped decode, fixup-into-ADD/RMS, D4 side stream, F32 `mul_mat_f` k-split/pair, DFlash2 concat quantizer.

---

# Appendix B: Gated DeltaNet and recurrent audit

## Headline findings

1. Every new prefill GDN kernel is gated off for a non-speculative context, independent of the arch gate. The builder only emits K>1 when `cparams.n_rs_seq > 0` (`delta-net-base.cpp:556,576`). With `n_rs_seq == 0` it calls `ggml_gated_delta_net(..., K=1)` (`:405`), so `keep_rs == false` (`gated_delta_net.cu:215`). L8 (`:66`), lazy rollback (`:222`), N1 chunked prefill (`:365`) and half-prefill (`:365`, `:393`) all require `keep_rs`.
2. The only ungated, directly applicable change is the batched warp reduction + hoisted `v` loads in the generic serial kernel (commit 6690be4), claimed bit-identical.
3. There is a chunked (WY-form) f32 prefill scan (N1, commit 239e6e8). Not bit-identical to the serial recurrence. Gated to sm_121, one sequence, S_v=128, keep_rs, T>=64, K<T.
4. "Half prefill" (C16, commit 91eb311) is a TF32/FP16 tensor-core variant of the chunked scan; rounds Q, K, the state operand, coefficients and the delta term to fp16 inside each 16-token chunk. Persistent state stays f32. Tests call it "intentionally numerical".
5. "Dedup" (GD2L) deduplicates per-head K/g/beta rollback records written once per column block. Verify-step only.
6. The ssm-conv rewrite is a fusion (GET_ROWS + CONCAT + snapshot CPYs + SSM_CONV + SILU + q/k L2_NORM), claimed and tested bit-exact; runs for prefill; gated to sm_121 and one sequence.
7. No measured numbers exist in the tree.

Baseline note: ca07598 already contains `cols_per_warp == 4` for all NVIDIA Ampere+ (`gated_delta_net_kernel.cuh:13`), the fused state gather (`s_ids`), the fused cache scatter, RAW gates, G_PRECOMPUTED on DGX Spark, and SSM_CONV+ADD+SILU fusion.

## T1. Batched warp reductions + hoisted v loads (generic serial kernel)

- `gdn_warp_reduce_sum_n<width,N>` runs one butterfly step for all 4 columns at a time (`gated_delta_net_kernel.cuh:17-28`); the 4 `v_t[col+c]` loads are issued before the reductions (`:408-412`). No gates beyond `cols_per_warp == 4`. Bit-identical. Commit 6690be4. Directly applicable.

## T2. L8 lane-permuted serial kernel with cp.async token staging

- 32 lanes split into 4 groups of 8; a column dot product needs 2 in-register adds + 3 shuffles instead of 5 shuffles per column: "6 shuffles per token" vs 40 (`:527,532`, `:556-563`). Token inputs staged into shared memory with `cp.async`, `GDN_L8_STAGE=4`, double buffered (`:673-722`). Absorbs a deferred zeroing SCALE on load.
- Gates `gated_delta_net.cu:66,105`: `keep_rs_t && !KDA`, S_v==128, `GGML_CUDA_GDN_L8` != 0, `cc == DGX_SPARK`, warp 32, `cols_per_warp == 4`, `n_tokens >= 17`. Multi-sequence capable (`blockIdx.y`, `s_ids[sequence]`, `:576,592`); K==1 writes slot 0 at the last token (`:822-833`).
- Claimed bit-identical ("same leaves in the same tree", `__fadd_rn`). Commits 6690be4, 923d628, 83d799c, 8539fa3, 239e6e8. Applicable after removing the sm_121 gate and the `keep_rs` branch.

## T3. Raw-gate activation pre-pass

- One kernel computes `exp(a*softplus(g+bias))` and `sigmoid(beta)` per (token, head) once (`:30-43`). Gates: raw, keep_rs, L8 on, S_v 128, `n_tokens >= 64`, DGX Spark. Bit-identical. Commit 6690be4.

## T4. N1: chunked WY-form f32 prefill (`gdn_chunk32_*`)

- 32-token chunks. `gdn_chunk32_prepare` computes per chunk decay products, `L_ij`, `M`, and two unit-lower triangular solves giving W and U (`:1213-1371`). `gdn_chunk32_scan` loops over chunks serially per head with the state tile in shared memory (`:1467-1551`, `:1553-1732`). Then the last K tokens run through the exact serial L8 kernel from the chunk boundary (`:2011-2018`). Variants `GGML_CUDA_GDN_SCAN_ILP`, `_PREFETCH`, `_WIDE`, `_COMPACT`, `_QSTAGE`; top switch `GGML_CUDA_GDN_CHUNKED`.
- Gates `gated_delta_net.cu:365-369`: `!kda && keep_rs && S_v == 128 && n_seqs == 1 && n_tokens >= 64 && K < n_tokens && cc == DGX_SPARK`. Kernels hard-wire one sequence (`s_ids[0]`, grid `dim3(H, 1, 8)`).
- Not bit-identical: test tolerance `atol = 2e-5, rtol = 2e-4` (`gdn_lazy_rollback_equivalence.cu:233-239`). Commit 239e6e8.
- Port: add a sequence grid dimension and per-sequence strides; for K==1 run the scan over all T tokens and keep its final-state store (`:1717-1731`) instead of the L8 tail.

## T5. C16 "half prefill"

- 16-token chunks, `H=32, HK=16, D=128` hard-coded (`gdn-half-prefill.cuh:7`); TF32 Gram, fp16 MMA operands, f32 state and output accumulation; final 16 tokens via exact L8. Gated inside N1 plus `GGML_CUDA_GDN_HALF_PREFILL`, `H==32 && neqk1==16 && K==16`. Approximate. Commits 91eb311, 2c62b6c. Not recommended as-is.

## T6. Packed head-major Q/K layout from the conv front

- Address permutation only; gate `ggml_cuda_gdn_l8_layout_eligible` (`gated_delta_net.cu:130-138`): DGX Spark, T>16, K>1. Commit 8539fa3. Only with T2/T4 and T10.

## T7–T9. Lazy rollback, L8V verify kernel, dedup/quotient/G1

- Speculative verify-step work; `n_seqs == 1`; irrelevant to prefill-only.

## T10. Conv front-end fusion (`ssm_conv_front_f32`, `ssm_conv_front_l2_short_f32`)

- One kernel at the CONCAT reads conv_states and x, writes the SILU output, the K conv-state snapshots, and the l2-normalized q/k heads; CONCAT, CPYs, SSM_CONV, SILU, L2_NORM are skipped (`ggml-cuda.cu:4760-4768`). Prefill path `n_t >= 64`. Gates `ggml-cuda.cu:4777` `cc != DGX_SPARK -> false`; `:4784` one sequence; d_conv 4. Exact by construction, `conv_front_equivalence.cpp` memcmp. Commits 923d628, 83d799c, 82ba412. Applicable after removing gates with a sequence grid dimension; small.

## T11. C68 ranged state clear

- GB10 only (`ggml_backend_cuda_track_writes`, `ggml-cuda.cu:1154`). Exact. Commit 83d799c.

## T12. Graph-order changes (`qwen35.cpp`, 8539fa3)

- `llm_schedule_prefill_gates`: `n_seqs == 1 && n_tokens > 16 && n_embd == 4096 && 32 heads`. Exact.

## Note on 4 sequences

- The GET_ROWS gather fusion matcher requires `ids->ne[0] == 1` and `v->ne[3] == 1` (`ggml-cuda.cu:4500,4513`, identical in baseline), so with 4 sequences each layer still runs a separate GET_ROWS copy of 4 x H x 64 KiB before the GDN op. Rows mode is disabled on CUDA.

---

# Appendix C: Attention, fusion and host-side audit

## Global

- Most CUDA-side changes are gated `cc == GGML_CUDA_CC_DGX_SPARK`. Shape constants are 9B-specific (`n_embd == 4096`, `n_head == 16`, `n_head_kv == 4`, head 256, GDN 32 x 128, Hadamard block 4096).
- CUDA graphs: `ggml_cuda_graph_check_compability` (ggml-cuda.cu:3170-3200) has no batch-size gate, so prefill-sized ubatches are graph-captured when the llama graph is reused.
- Attention with Q8_0 K/V: the in-place quantized-KV MMA path (`switch_ncols2<256,256,GGML_TYPE_Q8_0>`, fattn.cu:191) is baseline; no submission touched it. Both new FA kernels are F16-only.
- Harness prefill driver (engine_llama.cpp:360-381): one sequence, chunks of `n_batch`, `want_logits` only on the final token, F16 KV, FA on.

## 1. Last-attention-layer query narrowing

- At the last layer the graph gathers the single requested row of the layer input, replicates it 16x (`ggml_concat` x4, llama-last-attn.h:34-37), gathers the matching 4 MRoPE positions and the matching mask row (41-43), feeds those to the Q path and `build_attn` with the selected mask (`query_mask`, llama-graph.cpp:2954-2956), then keeps one of the 16 identical output rows (qwen35.cpp:450-455). K/V projections and cache writes still run on every row. 16 copies keep the matmul on the PQ2 J16 MMQ path instead of MMVQ. Baseline already narrowed the last layer's FFN (`narrow_last`).
- Activation qwen35.cpp:344-352: env `GGML_CUDA_LAST_ATTN_ROWS` (default on), `output_ids && il == n_layer-1 && n_outputs == 1 && ubatch.n_seqs_unq == 1 && flash_attn && n_embd_head == 256 && n_head == 16 && n_head_kv == 4 && wq,wo PQ2_0 && loras->empty() && !wq_s && !wo_s && !self_k_rot && !self_v_rot && !embeddings_layer_inp[n_layer]`; llama-last-attn.h:20-28 `n_seqs == 1`, `input->ne[0] == 4096`, `input->ne[1] > 16`, one output id, MROPE positions, F16 mask. No arch gate.
- Not bit-identical: test `prefill_last_attn_rows.cpp` requires K/V writes bit-exact and attention output within 2e-4 relative. Commits 8539fa3, 82ba412.

## 2. Prefill gate projections scheduled early

- Pure topological reorder. qwen35.cpp:495-497, llama-prefill-gates.h:12-25 (`n_seqs == 1`, `n_tokens > 16`, width 4096, weights `[4096,32]`). Bit-identical. Commit 8539fa3.

## 3. Device logits top-k (`llama_set_logits_topk`)

- `ggml_top_k` + `get_rows` on device; CUDA streaming selection (top-k.cu:76-108). Exact with a documented total order. Commits f685132 .. 6690be4. Low value for Shingi; the same `get_rows(reshape_3d(logits,...), ids)` pattern could gather fixed candidate ids on device.

## 4. Flash attention: shared-Q "lowstage" MMA kernel (fattn-lowstage.cuh)

- fattn.cu:68-83: `DKQ == DV == 128`, F16 KV, `cc == DGX_SPARK`, gqa 4, `Q->ne[1] == 16`, K in [512,1024], 8 KV heads. DFlash2 verify geometry. Decode-only. Commit 2c62b6c.

## 5. Flash attention: half-precision stream-K partials

- fattn-mma-f16.cuh:2113-2137: `DKQ=DV=256, ncols1=16, ncols2=4`, F16 KV, `cc == DGX_SPARK`, `Q->ne[1] == 16 && Q->ne[2] == 16`, K in [512,1024]. Bit-identical per `fattn_half_partials_equivalence.cu`. Decode-only. Commit 759629b.

## 6. RMS_NORM*w + MROPE/IMROPE fusion (+ K SET_ROWS, + V SET_ROWS)

- One block per (head, token) normalizes, applies rope_multi, optionally writes K/V rows (rope.cu:942-1066). Planner ggml-cuda.cu:6774-6830, env `GGML_CUDA_MROPE_FUSION`; `ggml_cuda_rms_norm_mul_mrope_supported` (rope.cu:1088-1101). K/V store variant requires SET_ROWS F16 or F32. No cc gate on the plan. Bit-identical per `attn_glue_equivalence.cu`. Commits 83d799c, 82ba412. Directly applicable for the norm+rope part; with Q8_0 cache the store part does not fuse.

## 7. Attention output gate folded into the FWHT/Q8 quantizer

- ggml-cuda.cu:6924-7050; `cc == DGX_SPARK`, Hadamard `n == 4096`, consumers PQ2 MMQ. Byte-exact. Commit 83d799c. Coupled to the GB10 D4 MMQ path.

## 8. Other FWHT/Q8 quantizer fusions

- `ggml_cuda_try_fwht_mmq` (ggml-cuda.cu:3578-4000): RMS fold (>16 rows), ADD+RMS (<=16 rows), SWIGLU, GDN group-norm+SWIGLU+permute. Gate `cc == DGX_SPARK`, `n == 4096`. Byte-exact tests. Commits 923d628, 83d799c, 8539fa3, f685132.

## 9. MMQ + residual ADD fusion and prefill GLU-pair GEMM

- `ggml_cuda_mmq_add_fusable` (ggml-cuda.cu:4385-4470): GB10, `mm->ne[1] > 16`, PQ2_0, MMQ path. `ggml_cuda_try_prefill_glu` (4005-4068). Both live inside the GB10 J=128 kernel.

## 10. SwiGLU float4 kernel

- unary.cu:281-296: `k` in [131072, 262144] elements. Decode-only by size window. Commit 2c62b6c.

## 11. ADD+RMS_NORM+MUL fusion: in-place alias relaxation

- `ggml_cuda_add_rms_norm_alias_ok` (ggml-cuda.cu:5578-5603) allows the allocator's in-place ADD. Any arch. Same kernel. Commits 6690be4, 923d628. Directly applicable.

## 12. PDL extended to glue kernels; memcpy replaced by a kernel

- `ggml_cuda_kernel_launch_gb10` (common.cuh:1937-1947); D2D copy kernel (cpy.cu:389-410). `cc == DGX_SPARK`. sm_89 irrelevant (PDL floor is Hopper); sm_120 marginal for prefill.

## 13. Per-request ranged state clear and asynchronous buffer clears

- KV high-water mark (`llama-kv-cache.cpp:1159`, 376-411); recurrent S write tracking GB10-only (ggml-cuda.cu:1155-1157); `ggml_cuda_zero_ranges_kernel` (1070-1145); async clears GB10-only (1037-1046). Env `LLAMA_STATE_CLEAR_RANGES`. Commit 83d799c. KV part arch-neutral.

## 14. CUDA-graph cache key and eviction policy

- `ggml_cuda_graph_get_key` mixes `n_nodes` and node0's shape (ggml-cuda.cu:3204-3215); eviction only above 32 cached graphs (common.cuh:1661-1663). Always on. Commits 8539fa3, 923d628. Directly applicable.

## 15. Graph slots and parked scheduler states

- `llama_set_graph_slots(n)` (llama-context.cpp:1754-1811). Irrelevant for a single-shape prefill service.

## 16. Scheduler input staging in pinned memory + CUDA upload side-stream

- ggml-backend.cpp:1611-1680; cap 4 MiB per split (`GGML_SCHED_MAX_STAGED_INPUTS`). Our F16 mask for 1024 x 8192 is about 16 MiB, so it falls back unless raised. Low value.

## 17. Token embedding table on the GPU

- engine_llama.cpp:200-206 overrides `^token_embd\.weight$` to the CUDA buffer type. Bit-identical per `token_embd_equivalence.cpp`. Commit 6690be4. Directly applicable; check whether Shingi already does this.

## 18. Boot warm-up and core pinning

- `boot_warmup` (main.cpp:68-100), `RESIDENT_BOOT_WARMUP` default 2. Commit 0870181. Applicable as a service pattern. Core pinning is GB10-specific.

## 19. Deferred zeroing SCALE of fresh recurrent cells

- `ggml_cuda_scale_deferrable` (ggml-cuda.cu:4346-4366), GB10. Marginal.

## 20. Decode / drafter-only items

- Decode gate side-stream, DOD QKV joint decode copy, decode fixup-add-rms, D4 side stream, DFlash2 device walk/rescore/conv/concat-Q8/head screen/embeddings import. All `n_tokens <= 16` or drafter contexts.
