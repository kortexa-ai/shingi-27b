"""One transport owner batches independent callers into a shared-weight native worker."""
import json
import platform
import queue
import selectors
import subprocess
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeout

from .gpu import gpu_free_mib, gpu_profile

CONTEXT_TOKENS = 16384
DEFAULT_PARALLEL = 1 if platform.system() == "Darwin" else 4
MAX_PENDING = 64


class NativeReadout:
    def __init__(self, executable, model, projector=None, *, parallel=DEFAULT_PARALLEL):
        if not isinstance(parallel, int) or not 1 <= parallel <= 4:
            raise ValueError("parallel must be 1 through 4")
        self.lock = threading.Lock()
        self.cache_state = {"entries": 0, "host_bytes": 0}
        self.pending = queue.Queue(MAX_PENDING)
        self.closed = threading.Event()
        self.worker = None
        _, preload, self.headroom = gpu_profile()
        if gpu_free_mib() < preload:
            raise RuntimeError(f"Shingi 27B requires at least {preload} MiB free on the GPU before loading")
        # The optional third argument is the vision projector; without it the readout is text only.
        command = [str(executable), str(model), str(CONTEXT_TOKENS)] + ([str(projector)] if projector else [])
        command += ["--parallel", str(parallel)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            self.info = self._read(300)
            if not self.info.get("ready"):
                raise RuntimeError("native model failed to initialize")
            if bool(projector) != bool(self.info.get("vision")):
                raise RuntimeError("native readout vision state does not match the requested projector")
            if gpu_free_mib() < self.headroom:
                raise RuntimeError("insufficient GPU headroom after model load")
            # Older/fake readouts retain the one-request protocol.
            self.parallel = min(parallel, int(self.info.get("parallel_slots", 1)))
            self.worker = threading.Thread(target=self._serve, name="shingi-readout", daemon=True)
            self.worker.start()
        except BaseException:
            self.close()
            raise

    def _read(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError("native readout timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("native readout exited")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("native readout returned malformed JSON") from exc

    @staticmethod
    def _result(result):
        if not isinstance(result, dict):
            raise RuntimeError("native readout returned a non-object")
        if "error" in result:
            error = ValueError if result.get("error_kind") == "input" else RuntimeError
            raise error(result["error"])
        return result

    @property
    def vision(self):
        return bool(self.info.get("vision"))

    @property
    def prefix_reuse(self):
        return bool(self.info.get("prefix_reuse"))

    @property
    def prefix_cache(self):
        """Limits and current host memory of the readout's cross-request prefix cache."""
        if not self.prefix_reuse:
            return None
        with self.lock:
            return {"max_entries": self.info.get("prefix_cache_entries"), "max_bytes": self.info.get("prefix_cache_bytes"),
                    **self.cache_state}

    def _serve(self):
        current = []
        try:
            while not self.closed.is_set():
                try:
                    first = self.pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                current = [first]
                # A small bounded admission window catches simultaneous HTTP callers.
                deadline = time.monotonic() + (0.002 if self.parallel > 1 else 0)
                while len(current) < self.parallel:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        current.append(self.pending.get(timeout=remaining))
                    except queue.Empty:
                        break
                if self.closed.is_set():
                    raise RuntimeError("native readout is closed")
                if gpu_free_mib() < self.headroom:
                    raise RuntimeError("GPU headroom fell below profile floor; native model stopped")
                if self.process.poll() is not None:
                    raise RuntimeError("native readout is not running")
                batched = len(current) > 1
                request = {"batch": [item[0] for item in current]} if batched else current[0][0]
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                result = self._read(max(item[2] for item in current))
                if not isinstance(result, dict):
                    raise RuntimeError("native readout returned a non-object")
                results = (result.get("responses") if "error" not in result else [result] * len(current)) if batched else [result]
                if not isinstance(results, list) or len(results) != len(current):
                    raise RuntimeError("native readout returned the wrong batch result count")
                for (_, future, _), value in zip(current, results):
                    try:
                        value = self._result(value)
                        if "cache" in value:
                            with self.lock:
                                self.cache_state = value["cache"]
                        future.set_result(value)
                    except ValueError as exc:
                        future.set_exception(exc)
                current = []
        except BaseException as exc:
            for _, future, _ in current:
                if not future.done():
                    future.set_exception(exc)
            self.close()

    def _call(self, request, timeout):
        future = Future()
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError("native readout is closed")
            try:
                self.pending.put_nowait((request, future, timeout))
            except queue.Full as exc:
                raise RuntimeError("native readout request queue is full") from exc
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            # Stop the exact owned worker: a timed-out pipe exchange cannot safely be reused.
            self.close()
            raise TimeoutError("native readout request timed out; process stopped") from exc

    def infer(self, prompt, labels, images=None):
        request = {"prompt": prompt, "labels": labels}
        if images:
            request["images"] = images
        return self._call(request, 300)

    def infer_prefix(self, prefix, suffixes, images=None, cache=True):
        """Evaluate the common prefix, then fork its state into independent question sequences.

        With cache=False the readout neither uses nor stores a cross-request prefix snapshot."""
        request = {"prefix": prefix, "suffixes": [{"text": text, "labels": labels} for text, labels in suffixes]}
        if images:
            request["images"] = images
        if not cache:
            request["cache"] = False
        return self._call(request, 300 + 10 * len(suffixes))

    def close(self):
        with self.lock:
            self.closed.set()
            while True:
                try:
                    _, future, _ = self.pending.get_nowait()
                except queue.Empty:
                    break
                if not future.done():
                    future.set_exception(RuntimeError("native readout is closed"))
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.worker and threading.current_thread() is not self.worker:
            self.worker.join(timeout=16)
