"""Apply the pinned device-state and SM120 recurrent-kernel corrections to an owned Prism checkout.

Prism's device writer/reader convert byte lengths into tensor dimensions. For Q8_0,
ggml_element_size is the size of a 32-element block, not one element. Both views
must multiply the block count by ggml_blck_size. Host snapshots are unaffected.
The capability symbol lets the readout fail closed with an unpatched runtime.
"""
import argparse
import subprocess
from pathlib import Path

REVISION = 'd8f26eec76da6d09bb708bcba51ef64b8cd868a3'
PATCH_ID = 'quantized-device-state-v1 gdn-columns-sm120-v1'
SOURCE = 'src/llama-context.cpp'
GDN_SOURCE = 'ggml/src/ggml-cuda/gated_delta_net.cu'


def patched(source):
    for item in ('winfo', 'rinfo'):
        old = f'const int64_t n = {item}.size/ggml_element_size({item}.tensor);'
        new = f'const int64_t n = ({item}.size/ggml_element_size({item}.tensor))*ggml_blck_size({item}.tensor->type);'
        if source.count(old) != 1:
            raise ValueError(f'unexpected pinned {item} device-state implementation')
        source = source.replace(old, new)
    return source + '\n// Shingi capability check: both device-state views preserve quantized block width.\nextern "C" LLAMA_API int shingi_prism_device_state_quantized_v1() { return 1; }\n'


def patched_gdn(source):
    # Match the host grid and device column ownership; Ada keeps its existing path.
    for old, new in (
        ('__CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK',
         '(__CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK || __CUDA_ARCH__ == GGML_CUDA_CC_BLACKWELL)'),
        ('cc == GGML_CUDA_CC_DGX_SPARK && S_v == 128',
         '(ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK || ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_BLACKWELL) && S_v == 128'),
    ):
        if source.count(old) != 1:
            raise ValueError('unexpected pinned recurrent column implementation')
        source = source.replace(old, new)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prism', type=Path)
    parser.add_argument('--check', action='store_true', help='verify without modifying')
    args = parser.parse_args()
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(args.prism), *argv], text=True).strip()
    if git('rev-parse', 'HEAD') != REVISION:
        raise SystemExit('Refusing an unexpected Prism revision')
    patches = {SOURCE: patched, GDN_SOURCE: patched_gdn}
    changed = git('diff', 'HEAD', '--name-only').splitlines()
    if any(name not in patches for name in changed):
        raise SystemExit('Refusing unrelated Prism source changes')
    pending = []
    for name, transform in patches.items():
        original = subprocess.check_output(['git', '-C', str(args.prism), 'show', f'{REVISION}:{name}']).decode()
        expected = transform(original)
        path = args.prism / name
        actual = path.read_text()
        if actual not in (original, expected):
            raise SystemExit(f'Refusing unknown changes in Prism source: {name}')
        if args.check and actual != expected:
            raise SystemExit(f'Prism patch is missing: {name}')
        if actual != expected:
            pending.append((path, expected))
    # Validate every file before changing any source, including an older managed checkout.
    if not args.check:
        for path, expected in pending:
            path.write_text(expected)
    print(PATCH_ID)


if __name__ == '__main__':
    main()
