"""Fixed-workload parallel readout checks; explicit paths, no service management or model changes."""
import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import socket
import copy
import httpx
import uvicorn
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from shingi.backend import NativeReadout
from shingi.decision import DecisionEngine, prefix_for, suffix_for
from shingi.gpu import gpu_snapshot
from shingi.server import create_app
from shingi.model import RUNTIME


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(report, reference, cases):
    if report["fixture_sha256"] != reference["fixture_sha256"]:
        raise RuntimeError("reference fixture differs")
    expected = {x["id"]: x["result"] for x in reference["answers"]}
    requests = {json.dumps(c["request"], sort_keys=True): c["id"] for c in cases}
    differences, mismatches = [], []
    def check(actual, identifier):
        wanted = expected[identifier]
        if set(actual["answers"]) != set(wanted["answers"]):
            raise RuntimeError("answer keys differ from reference")
        if actual["usage"]["input_tokens"] != wanted["usage"]["input_tokens"]:
            raise RuntimeError("logical input-token accounting differs")
        for key, a in wanted["answers"].items():
            b = actual["answers"][key]
            if a["type"] != b["type"]:
                raise RuntimeError("answer type differs")
            if a["type"] == "noul":
                u, v = [a["noul"], 1-a["noul"]], [b["noul"], 1-b["noul"]]
            else:
                if set(a["probabilities"]) != set(b["probabilities"]):
                    raise RuntimeError("candidate keys differ")
                u = [a["probabilities"][k] for k in sorted(a["probabilities"])]
                v = [b["probabilities"][k] for k in sorted(a["probabilities"])]
            if any(not math.isfinite(x) or not 0 <= x <= 1 for x in v) or abs(sum(v)-1) > 1e-6:
                raise RuntimeError("invalid probability distribution")
            differences.append(sum(abs(x-y) for x,y in zip(u,v))/2)
            if max(range(len(u)), key=u.__getitem__) != max(range(len(v)), key=v.__getitem__):
                mismatches.append({"id": identifier, "question": key, "expected": u, "actual": v})
    for result in report["answers"]:
        check(result["result"], result["id"])
    for run in report["runs"] + report.get("http", {}).get("runs", []):
        for result in run["results"]:
            check(result["result"], requests[json.dumps(result["request"], sort_keys=True)])
    for result in report.get("http", {}).get("mixed", []):
        identifier = requests.get(json.dumps(result["request"], sort_keys=True))
        if result["status"] == 200 and identifier:
            check(result["result"], identifier)
    summary = {"comparisons": len(differences), "mean_tvd": statistics.mean(differences),
               "max_tvd": max(differences), "winner_agreement": 1-len(mismatches)/len(differences),
               "mismatches": mismatches}
    report["comparison"] = summary
    if summary["mean_tvd"] > 0.005 or summary["max_tvd"] > 0.05 or summary["winner_agreement"] < 0.99:
        raise RuntimeError("serial/parallel probability agreement gate failed")


def http_checks(engine, cases, *, baseline=False):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(engine), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    report = {"runs": [], "errors": [], "mixed": []}
    try:
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("test HTTP server did not start")
            time.sleep(0.01)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
            report["version"] = client.get("/v1/version").json()
            work = [case["request"] for case in cases[:24]]
            def post(request):
                started = time.perf_counter()
                response = client.post("/v1/systemone", json=request)
                response.raise_for_status()
                return {"request": request, "result": response.json(), "wall_ms": (time.perf_counter()-started)*1000}
            for concurrency in [1, 2, 4, 4, 2, 1, 1, 4, 2]:
                started = time.perf_counter()
                with ThreadPoolExecutor(concurrency) as pool:
                    results = list(pool.map(post, work))
                report["runs"].append({"concurrency": concurrency, "seconds": time.perf_counter()-started, "results": results})
            # Interleave unrelated prefix states, an image, an oversized request and a
            # chunked choice. Admission errors must not contaminate healthy peers.
            mixed = [cases[24]["request"], cases[26]["request"], cases[27]["request"]]
            if engine.vision:
                mixed.append(cases[-1]["request"])
            oversized = copy.deepcopy(cases[0]["request"])
            oversized["state"] = "oversized input. " * 20000
            invalid = copy.deepcopy(cases[0]["request"])
            invalid["questions"] = {}
            literal = copy.deepcopy(cases[24]["request"])
            literal["state"] = "The literal text is <__media__>. This is not an image."
            literal["questions"] = {"literal": {"type": "noul", "instructions": "Is this explicitly described as text?"},
                                   "image": {"type": "noul", "instructions": "Is an actual image attached?"}}
            mixed += [oversized, invalid, literal]
            def mixed_post(request):
                response = client.post("/v1/systemone", json=request)
                return {"request": request, "status": response.status_code, "result": response.json()}
            with ThreadPoolExecutor(4) as pool:
                report["mixed"] = list(pool.map(mixed_post, mixed))
            # The released worker rejects a literal media marker without an image.
            # Record that known baseline defect; the candidate must accept the text.
            expected = [200] * (len(mixed)-3) + [422, 422, 422 if baseline else 200]
            if [r["status"] for r in report["mixed"]] != expected:
                raise RuntimeError("mixed valid/invalid HTTP status mismatch")
            report["recovery"] = post(cases[0]["request"])
            report["health"] = client.get("/health").json()
            report["generic"] = client.post("/v1/decisions", json={"input": "The color is red.", "questions": [
                {"id": "color", "type": "choice", "question": "What color is stated?", "options": [{"name": "red"}, {"name": "blue"}]},
                {"id": "red", "type": "yes_no", "question": "Is red stated?"}]}).json()
        return report
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("test HTTP server did not stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projector", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--baseline-backend", type=Path)
    parser.add_argument("--reference", type=Path, help="saved original-worker results; enforce frozen agreement gates")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--http", action="store_true", help="also exercise actual local HTTP endpoints and malformed-input recovery")
    args = parser.parse_args()
    fixture_path = Path(__file__).resolve().parents[1] / "tests/fixtures/parallel-decisions.json"
    fixture = json.loads(fixture_path.read_text())
    cases = fixture["cases"][:8] if args.canary else fixture["cases"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    report = {"fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
              "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "executable_sha256": file_sha256(args.executable),
              "model_sha256": file_sha256(args.model), "projector_sha256": file_sha256(args.projector) if args.projector else None,
              "runtime": RUNTIME, "calibration": {"temperature": 1.0, "noul_temperature": 1.0, "noul_bias": 0.0},
              "parallel": args.parallel, "baseline": bool(args.baseline_backend),
              "before": gpu_snapshot(), "answers": [], "runs": [], "memory": []}
    def save():
        (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    stop = threading.Event()
    def sample():
        while not stop.wait(0.2):
            report["memory"].append({"time": time.time(), **gpu_snapshot()})
    sampler = threading.Thread(target=sample, daemon=True)
    backend = None
    try:
        sampler.start()
        cls = NativeReadout
        kwargs = {"parallel": args.parallel}
        if args.baseline_backend:
            spec = importlib.util.spec_from_file_location("shingi.baseline_backend", args.baseline_backend)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls, kwargs = module.NativeReadout, {}
        backend = cls(args.executable, args.model, args.projector, **kwargs)
        report["info"] = backend.info
        report["loaded"] = gpu_snapshot()
        engine = DecisionEngine(backend)
        engine.evaluate(cases[0]["request"])
        for sample_case in cases:
            if sample_case["request"].get("images") and not args.projector:
                continue
            started = time.perf_counter()
            result, traces = engine.evaluate(sample_case["request"])
            report["answers"].append({"id": sample_case["id"], "result": result, "traces": traces,
                                      "wall_ms": (time.perf_counter() - started) * 1000})
            save()
        work = [case["request"] for case in cases[:8]]
        for concurrency in ([1, 4] if args.canary else [1, 2, 4, 4, 2, 1, 1, 4, 2]):
            def call(request):
                started = time.perf_counter()
                result, traces = engine.evaluate(request)
                return {"request": request, "result": result, "traces": traces,
                        "wall_ms": (time.perf_counter() - started) * 1000}
            started = time.perf_counter()
            with ThreadPoolExecutor(concurrency) as pool:
                results = list(pool.map(call, work))
            elapsed = time.perf_counter() - started
            report["runs"].append({"concurrency": concurrency, "seconds": elapsed, "results": results})
            print(json.dumps({"parallel": args.parallel, "concurrency": concurrency, "seconds": elapsed}), flush=True)
            save()
        if not args.canary:
            # Prefix replay and cache-off controls hold complete inputs and batch shape fixed.
            state = cases[24]["request"]["state"]
            prefix = prefix_for(state)
            suffixes = [(suffix_for("Is record zero red?", [("yes", ""), ("no", "")]), ["A", "B"])] * 8
            report["cache_controls"] = [backend.infer_prefix(prefix, suffixes, cache=cache) for cache in [False, True, True, False]]
            save()
            if args.http:
                report["http"] = http_checks(engine, cases, baseline=bool(args.baseline_backend))
                save()
        if args.reference:
            compare(report, json.loads(args.reference.read_text()), cases)
        report["finished"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if backend:
            backend.close()
        stop.set()
        sampler.join(timeout=12)
        report["after"] = gpu_snapshot()
        save()


if __name__ == "__main__":
    main()
