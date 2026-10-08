"""Concurrent callers must remain isolated across batching, input errors and worker failure."""
import json
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from shingi import backend as module

FAKE = '''
import json, sys, time
print(json.dumps({"ready": True, "vision": False, "parallel_slots": 4, "prefix_reuse": True}), flush=True)
for line in sys.stdin:
    packet = json.loads(line)
    batch = packet.get("batch", [packet])
    responses = []
    for r in batch:
        p = r.get("prompt", r.get("prefix"))
        if p == "slow": time.sleep(10)
        if p == "exit": sys.exit(0)
        if p == "malformed":
            print("not-json", flush=True)
            break
        if p == "bad":
            responses.append({"error": "bad input", "error_kind": "input"})
            continue
        result = {"logits": [float(p), -float(p)], "input_tokens": 2, "batch_size": len(batch)}
        if "prefix" in r:
            result = {"results": [result for s in r["suffixes"]], "prefix_tokens": 1}
        responses.append(result)
    else:
        print(json.dumps({"responses": responses} if "batch" in packet else responses[0]), flush=True)
'''


@pytest.fixture
def readout(tmp_path, monkeypatch):
    script = tmp_path / "readout"
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE))
    script.chmod(0o755)
    monkeypatch.setattr(module, "gpu_profile", lambda: (None, 0, 0))
    monkeypatch.setattr(module, "gpu_free_mib", lambda: 1 << 20)
    backend = module.NativeReadout(script, "model", parallel=4)
    yield backend
    backend.close()
    assert not backend.worker.is_alive()
    assert backend.process.poll() is not None


def test_independent_calls_are_batched_and_return_to_their_own_callers(readout):
    barrier = threading.Barrier(4)
    def call(i):
        barrier.wait(timeout=3)
        return readout.infer(str(i), ["A", "B"])
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(call, range(4)))
    assert [r["logits"] for r in results] == [[i, -i] for i in range(4)]
    assert max(r["batch_size"] for r in results) > 1


def test_input_error_does_not_kill_the_worker_or_other_requests(readout):
    barrier = threading.Barrier(4)
    def call(i):
        barrier.wait(timeout=3)
        if i == 1:
            with pytest.raises(ValueError, match="bad input"):
                readout.infer("bad", ["A", "B"])
            return None
        if i == 2:
            return readout.infer_prefix(str(i), [("q", ["A", "B"])])["results"][0]["logits"]
        return readout.infer(str(i), ["A", "B"])["logits"]
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(call, range(4))) == [[0, 0], None, [2, -2], [3, -3]]
    with pytest.raises(ValueError):
        readout.infer("bad", ["A", "B"])
    assert readout.infer("7", ["A", "B"])["logits"] == [7, -7]


@pytest.mark.parametrize("failure", ["exit", "malformed"])
def test_native_failure_releases_every_waiting_caller(readout, failure):
    barrier = threading.Barrier(4)
    def call(i):
        barrier.wait(timeout=3)
        try:
            return readout.infer(failure if i == 0 else str(i), ["A", "B"])
        except RuntimeError:
            return "failed"
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(call, range(4)))
    assert results[0] == "failed"
    with pytest.raises(RuntimeError):
        readout.infer("1", ["A", "B"])


def test_timeout_stops_owned_process_and_rejects_subsequent_work(readout):
    with pytest.raises(TimeoutError):
        readout._call({"prompt": "slow", "labels": ["A", "B"]}, 0.05)
    assert readout.process.poll() is not None
    with pytest.raises(RuntimeError, match="closed"):
        readout.infer("1", ["A", "B"])


def test_close_is_idempotent_and_wakes_pending_calls(readout):
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(readout.infer, "slow", ["A", "B"]) for _ in range(2)]
        readout.close()
        readout.close()
        for future in futures:
            with pytest.raises(RuntimeError):
                future.result(timeout=2)


def test_queue_is_bounded_and_overload_does_not_discard_admitted_work(readout, monkeypatch):
    import queue
    readout.pending = queue.Queue(2)
    entered, release = threading.Event(), threading.Event()
    def memory_check():
        entered.set()
        assert release.wait(timeout=3)
        return 1 << 20
    monkeypatch.setattr(module, "gpu_free_mib", memory_check)
    with ThreadPoolExecutor(3) as pool:
        first = pool.submit(readout.infer, "1", ["A", "B"])
        assert entered.wait(timeout=2)
        # Direct queue admission makes the overload boundary deterministic; the worker
        # is deliberately blocked at its pre-inference memory check.
        from concurrent.futures import Future
        pending = [Future(), Future()]
        for i, future in enumerate(pending):
            readout.pending.put_nowait(({"prompt": str(i + 2), "labels": ["A", "B"]}, future, 2))
        try:
            with pytest.raises(RuntimeError, match="queue is full"):
                readout.infer("9", ["A", "B"])
        finally:
            release.set()
        assert first.result(timeout=2)["logits"] == [1, -1]
        assert [f.result(timeout=2)["logits"] for f in pending] == [[2, -2], [3, -3]]
