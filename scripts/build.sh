#!/usr/bin/env bash
# Build the pinned Prism llama.cpp runtime, its multimodal library (mtmd) and Shingi's native
# readout into $SHINGI_HOME. Uses CUDA on Linux and Metal on macOS (Apple Silicon).
# Skips all work when the readout was already built from the same revision, backend, architectures and source.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHINGI_HOME="${SHINGI_HOME:-$HOME/.cache/shingi-27b}"
PRISM_URL=https://github.com/PrismML-Eng/llama.cpp.git
PRISM_REVISION=d8f26eec76da6d09bb708bcba51ef64b8cd868a3
ARCHITECTURES="${SHINGI_CUDA_ARCHITECTURES:-86;89;120;121}"
PRISM="$SHINGI_HOME/prism"
READOUT="$SHINGI_HOME/bin/readout"
STAMP="$READOUT.stamp"

VENDOR="${SHINGI_GPU_VENDOR:-cuda}"
if [ "$(uname -s)" = Darwin ]; then VENDOR=metal; fi
case "$VENDOR" in
    metal | cuda | rocm) ;;
    *) echo "shingi-27b: error: SHINGI_GPU_VENDOR must be cuda or rocm" >&2; exit 1 ;;
esac
if [ "$(uname -s)" = Darwin ]; then
    CHECKSUM="$(shasum -a 256 "$ROOT/src/native/readout.cpp" | cut -d' ' -f1)"
else
    CHECKSUM="$(sha256sum "$ROOT/src/native/readout.cpp" | cut -d' ' -f1)"
fi
case "$VENDOR" in
    metal)
        # Metal builds use their own build directory and stamp, so a CUDA build is never reused.
        BUILD="$PRISM/build-metal"
        stamp="$PRISM_REVISION Darwin-metal mtmd $CHECKSUM"
        ;;
    rocm)
        BUILD="$PRISM/build-rocm"
        stamp="$PRISM_REVISION rocm ${SHINGI_GPU_TARGETS:-gfx1100} mtmd $CHECKSUM"
        ;;
    cuda)
        BUILD="$PRISM/build"
        stamp="$PRISM_REVISION $ARCHITECTURES mtmd $CHECKSUM"
        ;;
esac
if [ -x "$READOUT" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$stamp" ]; then
    echo "shingi-27b: runtime already built at $PRISM_REVISION"
    exit 0
fi

case "$VENDOR" in
    metal) echo "shingi-27b: building the Prism runtime at $PRISM_REVISION for Metal" ;;
    rocm)
        echo "shingi-27b: building the Prism runtime at $PRISM_REVISION for ROCm targets ${SHINGI_GPU_TARGETS:-gfx1100}"
        command -v hipconfig >/dev/null 2>&1 || {
            echo "shingi-27b: error: hipconfig not found; install ROCm 6.1 or later" >&2
            exit 1
        }
        ;;
    cuda)
        echo "shingi-27b: building the Prism runtime at $PRISM_REVISION for CUDA architectures $ARCHITECTURES"
        echo "shingi-27b: the first build compiles GPU kernels and can take a while"
        ;;
esac
if [ ! -e "$PRISM" ]; then
    mkdir -p "$SHINGI_HOME"
    git clone --quiet --no-checkout "$PRISM_URL" "$PRISM"
fi
if [ "$(git -C "$PRISM" rev-parse HEAD 2>/dev/null)" != "$PRISM_REVISION" ]; then
    git -C "$PRISM" fetch --quiet origin "$PRISM_REVISION"
    git -C "$PRISM" checkout --quiet --detach "$PRISM_REVISION"
fi
if [ -n "$(git -C "$PRISM" status --porcelain --untracked-files=no)" ]; then
    echo "shingi-27b: $PRISM has local changes; move it away and run again" >&2
    exit 1
fi

if [ "$VENDOR" = rocm ]; then
    jobs="$(nproc)"
    [ "$jobs" -le 8 ] || jobs=8
    HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}"
    HIP_PATH="${HIP_PATH:-$(hipconfig -R)}"
    export HIPCXX HIP_PATH
    cmake -S "$PRISM" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_NATIVE=OFF \
        -DGPU_TARGETS="${SHINGI_GPU_TARGETS:-gfx1100}" -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_MTMD=ON -DMTMD_VIDEO=OFF
elif [ "$VENDOR" = metal ]; then
    jobs="$(sysctl -n hw.ncpu)"
    cmake -S "$PRISM" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_MTMD=ON -DMTMD_VIDEO=OFF
else
    jobs="$(nproc)"
    [ "$jobs" -le 8 ] || jobs=8  # CUDA kernel compilation needs several GB of RAM per job.
    cmake -S "$PRISM" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCHITECTURES" -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_MTMD=ON -DMTMD_VIDEO=OFF
fi
# mtmd is the image encoder library; the readout loads it only when a vision projector is given.
cmake --build "$BUILD" --target llama mtmd --parallel "$jobs"

mkdir -p "$SHINGI_HOME/bin"
c++ -std=c++17 -O2 -Wall -Wextra "$ROOT/src/native/readout.cpp" \
    -I"$PRISM/include" -I"$PRISM/ggml/include" -I"$PRISM/vendor" -I"$PRISM/tools/mtmd" \
    -L"$BUILD/bin" -Wl,-rpath,"$BUILD/bin" \
    -lmtmd -lllama -lggml -lggml-base -o "$READOUT"
printf '%s\n' "$stamp" > "$STAMP"
echo "shingi-27b: built $READOUT"
