#!/usr/bin/env bash
# Build, download and serve Shingi 27B with one command:
#   curl -fsSL https://raw.githubusercontent.com/kortexa-ai/shingi-27b/main/run.sh | bash
# Arguments after `bash -s --` (or after ./run.sh) are passed to the server.
set -euo pipefail

REPO_URL=https://github.com/kortexa-ai/shingi-27b.git
SHINGI_HOME="${SHINGI_HOME:-$HOME/.cache/shingi-27b}"
export SHINGI_HOME

say() { printf 'shingi-27b: %s\n' "$*" >&2; }
die() { say "error: $*"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found. $2"; }

# Print the checkout containing this script, or nothing when piped or copied elsewhere.
checkout_dir() {
    local source="${BASH_SOURCE[0]:-}" dir
    [ -n "$source" ] && [ -f "$source" ] || return 0
    dir="$(cd "$(dirname "$source")" && pwd)"
    if [ -f "$dir/src/native/readout.cpp" ] && grep -q '^name = "shingi-27b"' "$dir/pyproject.toml" 2>/dev/null; then
        printf '%s\n' "$dir"
    fi
}

# Clone or fast-forward the repository under $SHINGI_HOME and run its copy of this script.
enter_checkout() {
    local src="$SHINGI_HOME/src"
    need git "Install Git."
    if [ -d "$src/.git" ]; then
        say "updating $src"
        git -C "$src" pull --ff-only --quiet
    elif [ -e "$src" ]; then
        die "$src exists but is not a Git checkout; move it away or set SHINGI_HOME"
    else
        say "cloning $REPO_URL into $src"
        mkdir -p "$SHINGI_HOME"
        git clone --quiet "$REPO_URL" "$src"
    fi
    exec bash "$src/run.sh" "$@" </dev/null
}

VENDOR=""
gpu_vendor() {
    local vendor="${SHINGI_GPU_VENDOR:-cuda}"
    case "$vendor" in
        cuda | rocm) printf '%s\n' "$vendor" ;;
        *) die "SHINGI_GPU_VENDOR must be cuda or rocm" ;;
    esac
}

check_prerequisites() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    if [ "$os" = Darwin ]; then
        VENDOR=metal
        check_macos_prerequisites "$arch"
        return
    fi
    VENDOR="$(gpu_vendor)" || exit 1
    [ "$os" = Linux ] || die "Linux or macOS on Apple Silicon is required (found $os)"
    case "$arch" in
        x86_64 | aarch64) ;;
        *) die "x86-64 or aarch64 is required (found $arch)" ;;
    esac
    if [ "$VENDOR" = rocm ]; then
        if [ -d /opt/rocm/bin ]; then export PATH="/opt/rocm/bin:$PATH"; fi
        need hipconfig "Install ROCm 6.1 or later."
        need rocm-smi "Install ROCm 6.1 or later; it reports the AMD GPU memory."
    else
        if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
            export PATH="/usr/local/cuda/bin:$PATH"
        fi
        need nvidia-smi "Install the NVIDIA driver."
        need nvcc "Install the CUDA toolkit (12.9 or later) and put nvcc on PATH."
    fi
    need cmake "Install CMake 3.21 or later."
    need git "Install Git."
    need c++ "Install a C++17 compiler such as g++."
    check_cxx17_and_uv
}

# Apple Silicon only: the runtime uses Metal, and Intel Macs have no supported GPU path.
check_macos_prerequisites() {
    [ "$1" = arm64 ] || die "macOS requires Apple Silicon (arm64); Intel Macs are not supported (found $1)"
    # The Metal shaders are embedded and compiled at run time, so the Metal compiler is not needed.
    xcode-select -p >/dev/null 2>&1 || die "the Xcode command line tools are required; run xcode-select --install"
    need cmake "Install CMake 3.21 or later (for example: brew install cmake)."
    need git "Install Git (it comes with the Xcode command line tools)."
    need c++ "Install the Xcode command line tools: xcode-select --install"
    check_cxx17_and_uv
}

check_cxx17_and_uv() {
    printf 'int main() { return 0; }\n' | c++ -std=c++17 -x c++ - -o /dev/null >/dev/null 2>&1 \
        || die "c++ cannot compile C++17; install a newer compiler"
    if ! command -v uv >/dev/null 2>&1; then
        need curl "Install curl, or install uv yourself: https://docs.astral.sh/uv/"
        say "uv not found; installing it with the official installer (https://astral.sh/uv/install.sh) into ~/.local/bin"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        need uv "The uv installer did not put uv on PATH."
    fi
}

# Respect CUDA_VISIBLE_DEVICES; otherwise pick the GPU with the most free memory.
select_gpu() {
    local rows count best uuid free name
    if [ "$VENDOR" = rocm ]; then
        if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
            say "using ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
            return
        fi
        rocm-smi --showproductname >&2 || die "rocm-smi cannot list the AMD GPUs"
        die "set ROCR_VISIBLE_DEVICES to one index, for example: export ROCR_VISIBLE_DEVICES=0"
    fi
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        say "using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        return
    fi
    rows="$(nvidia-smi --query-gpu=uuid,memory.free,name --format=csv,noheader,nounits)"
    count="$(printf '%s\n' "$rows" | grep -c '^GPU-' || true)"
    [ "$count" -gt 0 ] || die "nvidia-smi lists no GPU"
    best="$(printf '%s\n' "$rows" | sort -s -t, -k2,2 -nr | head -n 1)"
    IFS=, read -r uuid free name <<<"$best"
    free="${free# }"
    name="${name# }"
    CUDA_VISIBLE_DEVICES="$uuid"
    export CUDA_VISIBLE_DEVICES
    case "$free" in
        '' | *[!0-9]*)
            # Unified-memory GPUs such as the DGX Spark GB10 report [N/A].
            if [ "$count" -eq 1 ]; then
                say "selected $name ($uuid); nvidia-smi does not report its memory, the server will check system memory"
            else
                say "selected $name ($uuid); nvidia-smi does not report memory, so this is the first listed GPU"
            fi
            ;;
        *) say "selected $name ($uuid) with $free MiB free according to nvidia-smi" ;;
    esac
}

main() {
    local dir
    dir="$(checkout_dir)"
    [ -n "$dir" ] || enter_checkout "$@"
    cd "$dir"
    check_prerequisites
    bash scripts/build.sh
    say "syncing the Python environment"
    uv sync --locked --no-dev --quiet
    # macOS has one Metal GPU; there is nothing to select.
    [ "$(uname -s)" = Darwin ] || select_gpu
    exec uv run --locked --no-dev shingi-27b --executable "$SHINGI_HOME/bin/readout" \
        --host "${SHINGI_HOST:-127.0.0.1}" --port "${SHINGI_PORT:-8765}" "$@"
}

main "$@"
