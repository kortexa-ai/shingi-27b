import functools
import json

import pytest
from shingi import gpu
from shingi.backend import NativeReadout

UUID = 'GPU-01234567-89ab-cdef-0123-456789abcdef'


@pytest.fixture(autouse=True)
def linux(monkeypatch):
    # The CUDA tests describe Linux; Darwin tests opt in with the `darwin` fixture.
    monkeypatch.setattr(gpu, 'is_macos', lambda: False)
    monkeypatch.delenv('SHINGI_GPU_VENDOR', raising=False)
    monkeypatch.delenv('ROCR_VISIBLE_DEVICES', raising=False)


@pytest.mark.parametrize('value', ['', '0', '0,1', 'GPU-short', UUID + ',' + UUID])
def test_requires_one_unambiguous_gpu(value, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', value)
    with pytest.raises(RuntimeError, match='exactly one full GPU UUID'):
        gpu.selected_gpu()


@pytest.mark.parametrize('total,expected', [(24564, (14*1024, 4*1024)), (97887, (30*1024, 10*1024))])
def test_profile_uses_capacity_without_machine_whitelist(total, expected, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: f'{UUID}, Test GPU, {total}, 18000\n')
    assert gpu.gpu_profile() == (UUID, *expected)
    assert gpu.gpu_free_mib() == 18000
    assert gpu.gpu_snapshot()['memory_source'] == 'nvidia-smi'


def test_rejects_wrong_device_response(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: 'GPU-other, Test GPU, 23028, 18000\n')
    with pytest.raises(RuntimeError, match='unexpected GPU identity'):
        gpu.gpu_snapshot()


def test_rejects_garbage_memory_values(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: f'{UUID}, Test GPU, lots, 18000\n')
    with pytest.raises(RuntimeError, match='invalid memory values'):
        gpu.gpu_snapshot()


MEMINFO = 'MemTotal:       128000000 kB\nMemFree:         9000000 kB\nMemAvailable:   100000000 kB\n'


@pytest.mark.parametrize('reported', ['[N/A], [N/A]', 'N/A, N/A', '[Not Supported], [Not Supported]'])
def test_unified_memory_falls_back_to_meminfo(reported, monkeypatch, tmp_path, capsys):
    # DGX Spark (GB10) reports [N/A] for memory; the GPU allocates from system memory.
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text(MEMINFO)
    monkeypatch.setattr(gpu, 'system_memory_mib', functools.partial(gpu.system_memory_mib, meminfo))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: f'{UUID}, NVIDIA GB10, {reported}\n')
    snapshot = gpu.gpu_snapshot()
    assert snapshot['total_mib'] == 128000000 // 1024 and snapshot['free_mib'] == 100000000 // 1024
    assert snapshot['memory_source'] == '/proc/meminfo MemAvailable'
    assert gpu.gpu_profile() == (UUID, 30 * 1024, 10 * 1024)
    assert '/proc/meminfo' in capsys.readouterr().out


def test_unreadable_meminfo_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match='does not report GPU memory'):
        gpu.system_memory_mib(tmp_path / 'missing')
    (tmp_path / 'partial').write_text('MemTotal: 1 kB\n')
    with pytest.raises(RuntimeError, match='does not report GPU memory'):
        gpu.system_memory_mib(tmp_path / 'partial')


def test_preload_gate_refuses_before_starting_the_runtime(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: f'{UUID}, Test GPU, 24564, 8000\n')
    started = []
    monkeypatch.setattr('shingi.backend.subprocess.Popen', lambda *a, **kw: started.append(a))
    with pytest.raises(RuntimeError, match='14336 MiB free'):
        NativeReadout('unused', 'unused')
    assert started == []


def test_driver_failure_is_backend_unavailability(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    def fail(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(gpu.subprocess, 'check_output', fail)
    with pytest.raises(RuntimeError, match='could not query'):
        gpu.gpu_snapshot()


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   100000.
Pages active:                                 900000.
Pages inactive:                               300000.
Pages speculative:                             50000.
Pages throttled:                                   0.
Pages wired down:                             200000.
Pages purgeable:                               10000.
"Translation faults":                    13132093240.
"""


def fake_macos(memsize, vm_stat=VM_STAT, calls=None):
    outputs = {('sysctl', '-n', 'hw.memsize'): f'{memsize}\n',
               ('sysctl', '-n', 'machdep.cpu.brand_string'): 'Apple M4 Pro\n',
               ('vm_stat',): vm_stat}
    def check_output(args, **kwargs):
        if calls is not None:
            calls.append(tuple(args))
        return outputs[tuple(args)]
    return check_output


@pytest.fixture
def darwin(monkeypatch):
    monkeypatch.setattr(gpu, 'is_macos', lambda: True)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)


@pytest.mark.parametrize('system,expected', [('Darwin', True), ('Linux', False)])
def test_platform_dispatch(system, expected, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(gpu.platform, 'system', lambda: system)
    assert gpu.is_macos() is expected


def test_darwin_memory_counts_free_inactive_speculative(darwin, monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'check_output', fake_macos(24 * 1024 ** 3))
    pages = 100000 + 300000 + 50000  # purgeable (10000) is excluded
    assert gpu.darwin_memory_mib() == (24 * 1024, pages * 16384 // 1024 ** 2)


def test_darwin_needs_no_uuid_or_nvidia_smi(darwin, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(gpu.subprocess, 'check_output', fake_macos(24 * 1024 ** 3, calls=calls))
    assert gpu.gpu_profile() == (None, 12 * 1024, 2 * 1024)
    assert not any(call[0] == 'nvidia-smi' for call in calls)
    snapshot = gpu.gpu_snapshot()
    assert snapshot['name'] == 'Apple M4 Pro'
    assert snapshot['memory_source'].startswith('vm_stat')
    assert 'vm_stat' in capsys.readouterr().out


def test_darwin_requires_16_gib(darwin, monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'check_output', fake_macos(8 * 1024 ** 3))
    with pytest.raises(RuntimeError, match='at least 16 GiB of unified memory'):
        gpu.gpu_profile()
    monkeypatch.setattr(gpu.subprocess, 'check_output', fake_macos(16 * 1024 ** 3))
    assert gpu.gpu_profile() == (None, 12 * 1024, 2 * 1024)


@pytest.mark.parametrize('vm_stat', ['', 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 1.\n',
                                     VM_STAT.replace('16384', 'many')])
def test_darwin_unreadable_memory_is_a_clear_error(vm_stat, darwin, monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'check_output', fake_macos(24 * 1024 ** 3, vm_stat))
    with pytest.raises(RuntimeError, match='sysctl and vm_stat'):
        gpu.gpu_snapshot()


def test_darwin_preload_gate_refuses_before_starting_the_runtime(darwin, monkeypatch):
    low = VM_STAT.replace('300000.', '1000.')  # about 2.5 GiB available
    monkeypatch.setattr(gpu.subprocess, 'check_output', fake_macos(24 * 1024 ** 3, low))
    started = []
    monkeypatch.setattr('shingi.backend.subprocess.Popen', lambda *a, **kw: started.append(a))
    with pytest.raises(RuntimeError, match='12288 MiB free'):
        NativeReadout('unused', 'unused')
    assert started == []


MIB = 1024 * 1024
VRAM_24GIB = 24560 * MIB

REAL_ROCM_SMI = json.dumps({'card0': {
    'VRAM Total Memory (B)': '8573157376', 'VRAM Total Used Memory (B)': '1217912832',
    'Card series': 'Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700 XT]',
    'Card model': 'Radeon RX 5600 XT',
    'Card vendor': 'Advanced Micro Devices, Inc. [AMD/ATI]', 'Card SKU': '1E4112U'}})

REAL_ROCM_SMI_NO_NAME = json.dumps(
    {'card0': {'VRAM Total Memory (B)': '8573157376',
               'VRAM Total Used Memory (B)': '1217912832'}})


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setenv('SHINGI_GPU_VENDOR', 'rocm')
    monkeypatch.setenv('ROCR_VISIBLE_DEVICES', '0')


def card(total, used=0, model='Radeon RX 7900 XTX'):
    return json.dumps({'card0': {'Card model': model, 'Card series': 'Navi 31',
                                 'VRAM Total Memory (B)': str(total),
                                 'VRAM Total Used Memory (B)': str(used)}})


def test_cuda_is_the_default_vendor(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    assert gpu.is_rocm() is False
    assert gpu.selected_gpu() == UUID


@pytest.mark.parametrize('value', ['rocm/intel', 'amd', 'CUDA', 'metal', ''])
def test_rejects_unknown_vendor(value, monkeypatch):
    monkeypatch.setenv('SHINGI_GPU_VENDOR', value)
    with pytest.raises(RuntimeError, match='SHINGI_GPU_VENDOR must be cuda or rocm'):
        gpu.is_rocm()


def test_metal_vendor_is_never_rocm(darwin, monkeypatch):
    monkeypatch.setenv('SHINGI_GPU_VENDOR', 'bogus')
    assert gpu.is_rocm() is False


@pytest.mark.parametrize('value', ['', ' ', '0,1', '0 1', 'GPU-DEADBEEFDEADBEEF',
                                   '7eff74a0-0000-1000-808f-7e20764e2714', '-0', '0x0'])
def test_rocm_requires_one_index(value, monkeypatch):
    monkeypatch.setenv('SHINGI_GPU_VENDOR', 'rocm')
    monkeypatch.setenv('ROCR_VISIBLE_DEVICES', value)
    with pytest.raises(RuntimeError, match='ROCR_VISIBLE_DEVICES to exactly one GPU index'):
        gpu.rocm_gpu()


def test_rocm_ignores_cuda_visible_devices(rocm, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    assert gpu.rocm_gpu() == '0'


def test_rocm_converts_bytes_to_mib(rocm, monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'check_output',
                        lambda *a, **kw: card(VRAM_24GIB, 800 * MIB))
    snapshot = gpu.gpu_snapshot()
    assert snapshot['total_mib'] == 24560
    assert snapshot['free_mib'] == 24560 - 800
    assert snapshot['uuid'] == '0'
    assert snapshot['memory_source'] == 'rocm-smi'
    assert snapshot['name'] == 'Radeon RX 7900 XTX'


def test_rocm_queries_rocm_smi(rocm, monkeypatch):
    seen = {}

    def record(*args, **kwargs):
        seen['argv'] = args[0]
        return card(VRAM_24GIB)

    monkeypatch.setattr(gpu.subprocess, 'check_output', record)
    gpu.gpu_snapshot()
    assert seen['argv'] == ['rocm-smi', '--showproductname', '--showmeminfo', 'vram', '--json']


def test_rocm_reads_real_rocm_smi_output(rocm, monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: REAL_ROCM_SMI)
    snapshot = gpu.gpu_snapshot()
    assert snapshot['name'] == 'Radeon RX 5600 XT'
    assert snapshot['total_mib'] == 8573157376 // MIB
    assert snapshot['free_mib'] == (8573157376 - 1217912832) // MIB


def test_rocm_starts_when_the_name_is_missing(rocm, monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: REAL_ROCM_SMI_NO_NAME)
    snapshot = gpu.gpu_snapshot()
    assert snapshot['name'] == 'AMD card 0'
    assert snapshot['total_mib'] == 8176


@pytest.mark.parametrize('payload', ['', 'not json', '{}', '{"card0": {}}',
                                     '{"card0": {"VRAM Total Memory (B)": "lots"}}',
                                     '{"card0": {"VRAM Total Memory (B)": "0"}}'])
def test_rocm_rejects_unusable_payload(payload, rocm, monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: payload)
    with pytest.raises(RuntimeError, match='no usable VRAM for card0'):
        gpu.gpu_snapshot()


def test_rocm_driver_failure_is_backend_unavailability(rocm, monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError('rocm-smi')

    monkeypatch.setattr(gpu.subprocess, 'check_output', fail)
    with pytest.raises(RuntimeError, match='could not query'):
        gpu.gpu_snapshot()


def test_rocm_profile_uses_the_shared_floor(rocm, monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: card(VRAM_24GIB))
    assert gpu.gpu_profile() == ('0', 14 * 1024, 4 * 1024)


def test_rocm_preload_gate_refuses_before_starting_the_runtime(rocm, monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: card(16368 * MIB))
    started = []
    monkeypatch.setattr('shingi.backend.subprocess.Popen', lambda *a, **kw: started.append(a))
    with pytest.raises(RuntimeError, match='at least 20 GiB'):
        NativeReadout('unused', 'unused')
    assert started == []
