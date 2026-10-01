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

- Linux on x86-64 or aarch64 with an NVIDIA or AMD GPU, or macOS on Apple Silicon
  (see [macOS](#macos) and [AMD](#amd-rocm)).
- An NVIDIA GPU with at least 20 GiB of memory. The model uses about 9.2 GiB with image input (8.2 GiB with `--no-vision`) at
  the full 16K context. Designed for RTX 4090 class 24 GB cards; the NVIDIA DGX Spark
  (GB10, unified memory) is supported.
  Tested on RTX PRO 6000, RTX 4090 and DGX Spark; speed is on the model card.
- Free memory at startup: 14 GiB on cards up to 32 GiB, 30 GiB on larger cards.
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

### AMD (ROCm)

Set `SHINGI_GPU_VENDOR=rocm` to build the runtime with HIP instead of CUDA. You
need ROCm 6.1 or later with `hipconfig` and `rocm-smi`, and a card with at least
20 GiB of VRAM; the free-memory figures above are the ones measured on NVIDIA.
Select the card with `ROCR_VISIBLE_DEVICES=<index from rocm-smi>`, because HIP
takes an index and AMD spells UUIDs differently between its own tools. The AMD
build is cached separately from the CUDA one, in `build-rocm`.

## Configuration

| Setting | Default | Change with |
|---|---|---|
| Port | `8765` | `--port N` or `SHINGI_PORT` |
| Host | `127.0.0.1` | `--host H` or `SHINGI_HOST` |
| GPU (Linux) | the GPU with the most free memory | `CUDA_VISIBLE_DEVICES=GPU-<full UUID from nvidia-smi -L>` |
| GPU backend (Linux) | `cuda` | `SHINGI_GPU_VENDOR=rocm` |
| AMD GPU | none selected automatically | `ROCR_VISIBLE_DEVICES=<index from rocm-smi>` |
| Model cache | the standard Hugging Face cache | `HF_HOME` (and `HF_TOKEN` if needed) |
| Model revision | `main` | `SHINGI_REVISION` |
| Build and checkout | `~/.cache/shingi-27b` | `SHINGI_HOME` |
| CUDA architectures (Linux) | `86;89;120;121` | `SHINGI_CUDA_ARCHITECTURES` |
| AMD GPU targets | `gfx1100` | `SHINGI_GPU_TARGETS` |
| Image input | on | `--no-vision` (text only, less memory) |

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
model ID, the GGUF SHA-256, the calibration, the runtime revision and whether
image input is loaded; `GET /health` reports readiness. The API is compatible
with the [TypeSafe](https://docs.typesafe.ai/api) System One primitives.

### Images

Add an `images` list to a request: up to 8 PNG, JPEG, GIF or BMP images, each
as base64 bytes or a `data:image/...;base64,` URL, at most 20 MiB each. They
are placed in order before the text, and every image uses at least 1,024 of
the 16,384 context tokens. Each question is a separate pass that reads the
images again, so every extra question about the same images costs a full
image pass. Text must not contain the `<__media__>` marker when images are
attached.

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
