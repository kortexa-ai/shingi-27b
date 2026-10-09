"""Apply only the pinned quantized-device-state correction to an owned Prism checkout.

Prism's device writer/reader convert byte lengths into tensor dimensions. For Q8_0,
ggml_element_size is the size of a 32-element block, not one element. Both views
must multiply the block count by ggml_blck_size. Host snapshots are unaffected.
The capability symbol lets the readout fail closed with an unpatched runtime.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

REVISION = 'd8f26eec76da6d09bb708bcba51ef64b8cd868a3'
PATCH_ID = 'quantized-device-state-v1'
SOURCE = 'src/llama-context.cpp'


def patched(source):
    for item in ('winfo', 'rinfo'):
        old = f'const int64_t n = {item}.size/ggml_element_size({item}.tensor);'
        new = f'const int64_t n = ({item}.size/ggml_element_size({item}.tensor))*ggml_blck_size({item}.tensor->type);'
        if source.count(old) != 1:
            raise ValueError(f'unexpected pinned {item} device-state implementation')
        source = source.replace(old, new)
    return source + '\n// Shingi capability check: both device-state views preserve quantized block width.\nextern "C" LLAMA_API int shingi_prism_device_state_quantized_v1() { return 1; }\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prism', type=Path)
    parser.add_argument('--check', action='store_true', help='verify without modifying')
    args = parser.parse_args()
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(args.prism), *argv], text=True).strip()
    if git('rev-parse', 'HEAD') != REVISION:
        raise SystemExit('Refusing an unexpected Prism revision')
    changed = git('diff', 'HEAD', '--name-only').splitlines()
    if any(name != SOURCE for name in changed):
        raise SystemExit('Refusing unrelated Prism source changes')
    original = subprocess.check_output(['git', '-C', str(args.prism), 'show', f'{REVISION}:{SOURCE}']).decode()
    expected = patched(original)
    path = args.prism / SOURCE
    actual = path.read_text()
    if actual not in (original, expected):
        raise SystemExit('Refusing unknown changes in Prism state I/O')
    if args.check and actual != expected:
        raise SystemExit('Prism quantized device-state patch is missing')
    if not args.check and actual != expected:
        path.write_text(expected)
    print(f'{PATCH_ID} {hashlib.sha256(expected.encode()).hexdigest()}')


if __name__ == '__main__':
    main()
