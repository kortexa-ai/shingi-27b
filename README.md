![Shingi — 審議](assets/banner.png)

# Shingi 27B

Shingi 27B is a local decision model. Give it context (text and, optionally,
images), a question and the possible answers; it returns a choice, calibrated
probabilities or an ordinal score. It reads every candidate answer's logit directly and generates no text,
so there is nothing to parse. The model is Bonsai 2 27B (ternary PQ2_0) with
trained per-block scales, served through the Prism llama.cpp runtime.

## Quickstart

```bash
curl -fsSL https://raw.githubusercontent.com/kortexa-ai/shingi-27b/main/run.sh | bash
```

or

```bash
git clone https://github.com/kortexa-ai/shingi-27b.git
cd shingi-27b
./run.sh
```

The first run builds the pinned Prism runtime and the readout, sets up the
Python environment, downloads the model (about 7.2 GB) and the Bonsai 2 27B
vision projector (about 0.6 GB), verifies their SHA-256, and starts the API on
`http://127.0.0.1:8765`. Later runs reuse all of that.

## Requirements

- Linux on x86-64 or aarch64 with an NVIDIA GPU, or macOS on Apple Silicon
  (see [macOS](#macos)).
- An NVIDIA GPU with at least 20 GiB of memory. Four-sequence serving with image input
  used about 9.8 GiB in the RTX PRO 6000 checks at the full 16K context; memory depends
  on the device, sequence count and workload. Designed for RTX 4090 class 24 GB cards; the NVIDIA DGX Spark
  (GB10, unified memory) is supported.
  Tested on RTX PRO 6000, RTX 4090 and DGX Spark; speed is on the model card.
- Free memory at startup: 14 GiB on cards up to 32 GiB, 30 GiB on larger cards.
  Add the configured VRAM cache budget (1 GiB by default) to those startup floors.
  At least 4 GiB (10 GiB on larger cards) must stay free while serving.
- NVIDIA driver, CUDA toolkit 12.9 or later (`nvcc`), CMake, a C++17 compiler
  and Git. `run.sh` installs [uv](https://docs.astral.sh/uv/) if it is missing.
- About 8 GB of disk for the weights, plus space for the runtime build.

### macOS

Apple Silicon Macs run the same runtime with Metal. 24 GB of unified memory or
more is recommended; 16 GB is the minimum. At startup 12 GiB must be free, and
2 GiB must stay free while serving. You need the Xcode command line tools
(`xcode-select --install`), CMake and Git. Intel Macs are not supported.

Macs are much slower than CUDA GPUs. On an M4 Pro (64 GB) the examples below
(about 70 tokens) take roughly 1.2 s each and a request of about 1,100 tokens
roughly 12.6 s. The model uses about 8–9 GB at the full 16K context.

## Configuration

| Setting | Default | Change with |
|---|---|---|
| Port | `8765` | `--port N` or `SHINGI_PORT` |
| Host | `127.0.0.1` | `--host H` or `SHINGI_HOST` |
| GPU (Linux) | the GPU with the most free memory | `CUDA_VISIBLE_DEVICES=GPU-<full UUID from nvidia-smi -L>` |
| Model cache | the standard Hugging Face cache | `HF_HOME` (and `HF_TOKEN` if needed) |
| Model revision | the pinned release commit | `SHINGI_REVISION` |
| Build and checkout | `~/.cache/shingi-27b` | `SHINGI_HOME` |
| CUDA architectures (Linux) | `86;89;120;121` | `SHINGI_CUDA_ARCHITECTURES` |
| Image input | on | `--no-vision` (text only, less memory) |
| Parallel sequences | 4 on CUDA, 1 on macOS | `--parallel 1` through `--parallel 4` |
| Prefix cache location | `vram` on CUDA, `host` on macOS | `--prefix-cache vram`, `host` or `off` |
| Prefix cache budget | 1024 MiB | `--prefix-cache-mib N` (0 disables caching) |

Pass arguments through the one-liner with `bash -s --`:

```bash
curl -fsSL https://raw.githubusercontent.com/kortexa-ai/shingi-27b/main/run.sh | bash -s -- --port 9000
```

The API has no authentication. Keep it on localhost unless you put your own
access control in front of it.

If you already have a readout built against the pinned Prism runtime, run the
server directly. An omitted model or calibration is downloaded from the Hub:

```bash
uv run shingi-27b --executable PATH [--model PATH] [--calibration PATH] [--mmproj PATH] [--port N]
```

`--skip-verify` skips the SHA-256 checks for people who know what they are doing.

## API

`POST /v1/systemone` takes a `state`, and named `questions` of type `choice`
(1–255 named options), `noul` (yes/no) or `score` (2–10 ordered levels).

A choice:

```bash
curl -s http://127.0.0.1:8765/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"shingi-27b","state":"Please replace my damaged parcel.","questions":{"route":{"type":"choice","instructions":"Select the handling team.","criteria":{"returns":"Damaged goods and replacements","billing":"Charges and refunds","shipping":"Delivery tracking"}}}}'
```

The answer has the selected `choice`, `probabilities` for every option and a
`confidence`.

A yes/no question:

```bash
curl -s http://127.0.0.1:8765/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"shingi-27b","state":"Please replace my damaged parcel.","questions":{"refund":{"type":"noul","instructions":"Is the customer asking for their money back?"}}}'
```

The answer's `noul` is the probability of yes. `GET /v1/version` reports the
model ID, the GGUF SHA-256, the calibration, the runtime revision, whether
image input is loaded, the parallel sequence count and the size of the prefix cache; `GET /health` reports
readiness. The API is compatible
with the [TypeSafe](https://docs.typesafe.ai/api) System One primitives.

### Parallel requests and shared state

The CUDA worker batches concurrent callers and questions from the same request into
up to four independent sequences. They share one loaded model and a 16,384-token
KV-memory pool. Identical cached prefixes can share their device state; unrelated
requests keep separate state. A long request can use the whole context pool, so fewer
sequences run together when their combined tokens would exceed it. Oversized prompts
still return an error instead of being truncated.

The Python transport admits at most 64 waiting calls and gives simultaneous callers
a 2 ms batching window. Keeping twice the sequence count in flight (eight requests
for four slots) can fill the next batch while the current one runs. Measure the actual
workload: with bounded tail padding, eight callers were slightly faster in Mappity
on both tested CUDA cards; an earlier engine favored four. Queue overflow returns 503. Image encoding runs serially;
questions about a shared image prefix can then run together. `--parallel 1` retains
the serial comparison path. macOS defaults to that path; CUDA measurements do not
establish parallel Metal performance.

Different batch shapes can slightly change floating-point scores. See the
[parallel inference checks](results/parallel-decisions/REPORT.md) for correctness,
probability differences, latency, memory and reproducible commands. More slots do not
guarantee a speedup for every workload. Model weights and calibration are unchanged.

A separate [6000 sequence-scaling study](results/sequence-scaling/REPORT.md) tested
4–32 independent sequences. Larger batches did not remove the GPU computation
bottleneck; its experimental patch leaves production defaults unchanged.

The CUDA serving memory guard reads NVML directly on every batch, using the selected
GPU UUID. It retains the driver handle but never caches free-memory readings.
Startup inventory still uses `nvidia-smi`; unified-memory GPUs retain their system
memory check. Driver errors stop the worker. Native parallel traces measure decode
time through CUDA synchronization. Per-question `prefill_ms` allocates each decode
call across only the sequences it evaluated, plus their prefix restore share.
`decode_ms` and `decode_calls` describe the whole native exchange and are repeated
in its responses; sum them only once per exchange. These are elapsed wall timings,
not CUDA-event kernel timings. `NativeReadout.batch_counts` reports completed
exchanges by caller count for occupancy diagnostics.
Parallel mode uses up to 1024 tokens per decode call. For unequal sequence lengths,
the worker can append at most 128 discarded tail tokens per call when the shared
context has room. Logits are read at each real prompt end, before that padding.
This reduces small remainder calls without changing the model inputs used for an
answer. `padding_tokens` is an exchange total; logical `usage.input_tokens`
excludes it. The serial path retains 512-token calls.

### Prefix cache

Requests with several questions can reuse the exact same state and images across
requests. The CUDA default keeps these snapshots in VRAM through Prism's on-device
state API. Tensor payloads stay on the GPU; small metadata and cache keys stay in
host memory. Short prefixes can also be cached. This is exact-prefix matching,
not a partial-prefix or radix cache. Single-question requests use the direct
inference path and do not populate this cache.

The cache has four ownership slots and a conservative byte budget. A snapshot in
use cannot be evicted. Prism retains a slot's GPU allocation after its logical
entry expires, so the budget counts retained allocations as well as live entries.
When a snapshot does not fit, the worker recomputes the prompt without spilling
state to host RAM. A larger budget permits larger snapshots, not more slots.
Model weights, working KV/recurrent state and compute buffers are separate from
this cache budget. Startup reserves the budget in addition to the memory floors.

Use `--prefix-cache host --prefix-cache-mib 1024` to opt into host snapshots and
CPU/GPU transfers, or `--prefix-cache off` to recompute prefixes. Host mode saves
complete text blocks and avoids snapshots for short text prefixes. Neither mode
is an automatic secondary tier. `/v1/version` reports the mode, limits, retained
device reservation, host cache bytes, hits, misses, evictions and bypasses.

The build applies a narrow correction to the pinned Prism runtime's quantized
device-state views. The worker refuses VRAM mode without that runtime capability;
`/v1/version` lists `quantized-device-state-v1` under runtime patches. The patch
is confined to Shingi's owned runtime checkout. It does not change model weights
or calibration.
See the [device-cache checks](results/vram-cache/REPORT.md) for correctness,
memory bounds, matched transfer timings and full Mappity measurements.

### Images

Add an `images` list to a request: up to 8 PNG, JPEG, GIF or BMP images, each
as base64 bytes or a `data:image/...;base64,` URL, at most 20 MiB each. They
are placed in order before the text, and every image uses at least 1,024 of
the 16,384 context tokens. The questions of one request share one pass over
the images and the state. The runtime forks a saved prefix into independent question
sequences. A sole shared prefix stays on the GPU across waves of questions.
The prefix cache can skip image encoding in a later request with identical images
and state. Each question still requires evaluation; snapshot, restore and scheduling
costs are included in measured latency. `usage`
counts the image tokens once per request. Text must not contain the
`<__media__>` marker when images are attached.

```bash
IMAGE=$(base64 < photo.png | tr -d '\n')
curl -s http://127.0.0.1:8765/v1/systemone -H 'Content-Type: application/json' -d @- <<JSON
{"model":"shingi-27b","state":"A photo from the loading dock.","images":["$IMAGE"],
 "questions":{"damaged":{"type":"noul","instructions":"Is the parcel damaged?"}}}
JSON
```

### SGLang decisions

`POST /v1/decisions` accepts the request shapes of SGLang's
[decision route](https://docs.sglang.io/docs/supported-models/decision_models),
so SGLang clients can use Shingi by changing the base URL. A body with an
`input` and a list of `choice`, `score` and `yes_no` questions returns the
generic answers (`probabilities`, `choice` or `score`, `label_mass`); a body
with a `state` and named `choice`, `score` and `noul` questions returns System
One answers with `decision`, as SGLang serves decision models. Both accept
`images` and an optional `temperature`. Shingi uses its own prompt for both.

```bash
curl -s http://127.0.0.1:8765/v1/decisions -H 'Content-Type: application/json' -d @- <<JSON
{"input":"The integration keeps failing and I am losing sales.",
 "questions":[{"id":"urgent","type":"yes_no","question":"The customer needs an answer today."},
              {"id":"team","type":"choice","question":"Which team should handle this?",
               "options":[{"name":"billing"},{"name":"technical"},{"name":"sales"}]}]}
JSON
```

## Model card

Training, evaluation results and limitations are on the
[model card](https://huggingface.co/kortexa-ai/shingi-27b).

## License

The code is Apache-2.0 (see [LICENSE](LICENSE)). The weights derive from Bonsai
2 27B by Prism ML (Apache-2.0), and image input uses its vision projector
unchanged; the Prism runtime is MIT. See [NOTICE](NOTICE).
