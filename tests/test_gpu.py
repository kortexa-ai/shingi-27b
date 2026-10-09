import functools

import pytest
from shingi import gpu
from shingi.backend import NativeReadout

UUID = 'GPU-01234567-89ab-cdef-0123-456789abcdef'


@pytest.fixture(autouse=True)
def linux(monkeypatch):
    # The CUDA tests describe Linux; Darwin tests opt in with the `darwin` fixture.
    monkeypatch.setattr(gpu, 'is_macos', lambda: False)


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
    monkeypatch.setattr('shingi.backend.gpu_free_mib', lambda: 8000)
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


class NvmlFunction:
    def __init__(self, work):
        self.work = work
    def __call__(self, *args):
        return self.work(*args)


@pytest.fixture
def nvml(monkeypatch):
    from types import SimpleNamespace
    state = SimpleNamespace(free=18000, total=24564, status=0, uuid=None, loads=0, queries=0, shutdowns=0)
    def handle(uuid, out):
        state.uuid = uuid.decode()
        out._obj.value = 123
        return 0
    def query(handle, out):
        assert handle.value == 123
        state.queries += 1
        out._obj.total = state.total * 1024**2
        out._obj.free = state.free * 1024**2
        out._obj.used = 0
        return state.status
    def shutdown():
        state.shutdowns += 1
        return 0
    lib = SimpleNamespace(nvmlInit_v2=NvmlFunction(lambda: 0), nvmlShutdown=NvmlFunction(shutdown),
                          nvmlDeviceGetHandleByUUID=NvmlFunction(handle), nvmlDeviceGetMemoryInfo=NvmlFunction(query))
    def load(name):
        assert name == "libnvidia-ml.so.1"
        state.loads += 1
        return lib
    gpu._nvml_memory.cache_clear()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.ctypes, 'CDLL', load)
    monkeypatch.setattr(gpu.atexit, 'register', lambda fn: None)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: pytest.fail('memory guard spawned a process'))
    yield state, lib
    gpu._nvml_memory.cache_clear()


def test_nvml_reuses_handle_but_reads_fresh_memory(nvml):
    state, _ = nvml
    assert gpu.gpu_free_mib() == 18000
    state.free = 900
    assert gpu.gpu_free_mib() == 900
    assert (state.loads, state.queries, state.uuid) == (1, 2, UUID)


def test_nvml_handle_tracks_selected_uuid(nvml, monkeypatch):
    state, _ = nvml
    gpu.gpu_free_mib()
    other = 'GPU-11111111-89ab-cdef-0123-456789abcdef'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', other)
    gpu.gpu_free_mib()
    assert state.uuid == other and state.loads == 2


@pytest.mark.parametrize('status', [4, 15, 999])
def test_nvml_driver_failures_do_not_return_stale_memory(nvml, status):
    state, _ = nvml
    assert gpu.gpu_free_mib() == 18000
    state.status = status
    with pytest.raises(RuntimeError, match='NVML error'):
        gpu.gpu_free_mib()


def test_nvml_not_supported_uses_fresh_system_memory(nvml, monkeypatch):
    state, _ = nvml
    state.status = 3
    monkeypatch.setattr(gpu, 'system_memory_mib', lambda: (128000, 99000))
    assert gpu.gpu_free_mib() == 99000
    monkeypatch.setattr(gpu, 'system_memory_mib', lambda: (128000, 1000))
    assert gpu.gpu_free_mib() == 1000


def test_nvml_rejects_invalid_memory(nvml):
    state, _ = nvml
    state.free = 30000
    with pytest.raises(RuntimeError, match='invalid memory'):
        gpu.gpu_free_mib()


def test_nvml_unavailable_is_a_clear_error(monkeypatch):
    gpu._nvml_memory.cache_clear()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    def missing(*args):
        raise OSError('missing driver library')
    monkeypatch.setattr(gpu.ctypes, 'CDLL', missing)
    with pytest.raises(RuntimeError, match='NVIDIA memory monitor'):
        gpu.gpu_free_mib()
