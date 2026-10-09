"""Cache configuration and memory admission, without loading model weights."""
import json
import sys

import pytest

from shingi import backend


@pytest.mark.parametrize('mode,mib', [('unknown', 1024), ('vram', -1), ('host', 65537), ('off', True), ('vram', 1.5)])
def test_invalid_cache_policy_fails_before_gpu_access(mode, mib, monkeypatch):
    monkeypatch.setattr(backend, 'gpu_profile', lambda: pytest.fail('GPU accessed before validation'))
    with pytest.raises(ValueError):
        backend.NativeReadout('unused', 'unused', prefix_cache=mode, prefix_cache_mib=mib)


def test_vram_reservation_is_included_in_preload_gate(monkeypatch):
    monkeypatch.setattr(backend, 'gpu_profile', lambda: (None, 1000, 100))
    monkeypatch.setattr(backend, 'gpu_free_mib', lambda: 1500)
    monkeypatch.setattr(backend.subprocess, 'Popen', lambda *a, **k: pytest.fail('unsafe model launch'))
    with pytest.raises(RuntimeError, match='1024 MiB for the VRAM cache'):
        backend.NativeReadout('unused', 'unused', prefix_cache='vram', prefix_cache_mib=1024)


@pytest.mark.parametrize('mode,mib,expected', [('vram', 256, 'vram'), ('host', 256, 'host'), ('off', 256, 'off'), ('vram', 0, 'off')])
def test_policy_reaches_worker_and_version(tmp_path, monkeypatch, mode, mib, expected):
    executable = tmp_path / 'readout'
    executable.write_text(f'''#!{sys.executable}
import sys,json
args=dict(zip(sys.argv[-6::2],sys.argv[-5::2]))
print(json.dumps({{"ready":True,"vision":False,"prefix_reuse":True,"parallel_slots":1,
"prefix_cache_mode":args["--prefix-cache"],"prefix_cache_bytes":int(args["--prefix-cache-mib"])*1048576,
"prefix_cache_entries":4,"device_state_patch":"quantized-device-state-v1"}}),flush=True)
for line in sys.stdin:
 print(json.dumps({{"logits":[0,1],"cache":{{"mode":args["--prefix-cache"],"entries":1,"host_bytes":123,"device_reserved_bytes":167772160}}}}),flush=True)
''')
    executable.chmod(0o755)
    monkeypatch.setattr(backend, 'gpu_profile', lambda: (None, 0, 0))
    monkeypatch.setattr(backend, 'gpu_free_mib', lambda: 1 << 20)
    worker = backend.NativeReadout(executable, 'model', parallel=1, prefix_cache=mode, prefix_cache_mib=mib)
    try:
        assert worker.prefix_cache['mode'] == expected
        assert worker.prefix_cache['max_bytes'] == (0 if expected == 'off' else mib * 1048576)
        worker.infer('prompt', ['A', 'B'])
        assert worker.prefix_cache['device_reserved_bytes'] == 167772160
        assert worker.prefix_cache['host_bytes'] == 123
        from fastapi.testclient import TestClient
        from shingi.decision import DecisionEngine
        from shingi.server import create_app
        with TestClient(create_app(DecisionEngine(worker))) as client:
            version = client.get('/v1/version').json()
        assert version['runtime']['patches'] == ['quantized-device-state-v1']
    finally:
        worker.close()


def test_runtime_patch_refuses_unknown_edits_and_is_idempotent(tmp_path, monkeypatch, capsys):
    import importlib.util
    import subprocess
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('patch_prism', Path(__file__).resolve().parents[1] / 'scripts/patch-prism.py')
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    root = tmp_path / 'prism'
    (root / 'src').mkdir(parents=True)
    source = root / 'src/llama-context.cpp'
    original = '\n'.join(f'const int64_t n = {item}.size/ggml_element_size({item}.tensor);' for item in ('winfo', 'rinfo'))+'\n'
    source.write_text(original)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    git('init', '--quiet')
    git('add', 'src/llama-context.cpp')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '--quiet', '-m', 'fixture')
    monkeypatch.setattr(patcher, 'REVISION', git('rev-parse', 'HEAD'))
    monkeypatch.setattr(sys, 'argv', ['patch-prism', str(root)])
    patcher.main()
    assert source.read_text() == patcher.patched(original)
    patcher.main()
    assert source.read_text() == patcher.patched(original)
    monkeypatch.setattr(sys, 'argv', ['patch-prism', str(root), '--check'])
    patcher.main()
    source.write_text(source.read_text()+'unexpected edit\n')
    with pytest.raises(SystemExit, match='unknown changes'):
        patcher.main()
    source.write_text(original)
    with pytest.raises(SystemExit, match='patch is missing'):
        patcher.main()
