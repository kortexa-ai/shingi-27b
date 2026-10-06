"""Shared-prefix reuse: one readout call per request round, with the same answers as separate prompts."""
import hashlib
import json
import math
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from shingi import backend as backend_module
from shingi.decision import MEDIA_MARKER, DecisionEngine, prefix_for, prompt_for, suffix_for
from shingi.server import create_app

PNG_B64 = "iVBORw0KGgoAAAAAAAAAAAAAAAAAAAAA"


def logits_for(prompt, labels):
    """Deterministic logits that depend on the whole prompt, so any prompt difference shows."""
    return [int(hashlib.sha256(f"{prompt}|{label}".encode()).hexdigest()[:8], 16) / 2 ** 32 * 6 for label in labels]


class PromptReadout:
    """Answers single prompts; with prefix_reuse it also answers prefix calls with the same logits."""

    def __init__(self, prefix_reuse=True, vision=True):
        self.prefix_reuse = prefix_reuse
        self.vision = vision
        self.calls = []

    def infer(self, prompt, labels, images=None):
        self.calls.append(("infer", prompt, labels, images))
        result = {"logits": logits_for(prompt, labels), "input_tokens": len(prompt), "prefill_ms": 1.0,
                  "log_normalizer": 10.0}
        if images:
            result |= {"images": len(images), "image_tokens": 1024 * len(images)}
        return result

    def infer_prefix(self, prefix, suffixes, images=None):
        self.calls.append(("prefix", prefix, suffixes, images))
        results = [{"logits": logits_for(prefix + text, labels), "input_tokens": len(prefix + text),
                    "suffix_tokens": len(text), "prefill_ms": 0.5, "log_normalizer": 10.0} for text, labels in suffixes]
        response = {"results": results, "prefix_tokens": len(prefix), "prefix_ms": 4.0}
        if images:
            response |= {"images": len(images), "image_tokens": 1024 * len(images)}
        return response


QUESTIONS = {
    "route": {"type": "choice", "instructions": "route?", "criteria": {"x": None, "y": "why", "z": ["zed"]}},
    "yes": {"type": "noul", "instructions": "yes?"},
    "rating": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", {"level": "high"}]},
    "many": {"type": "choice", "instructions": "pick many", "criteria": {f"o{i:03}": f"option {i}" for i in range(120)}},
}


def evaluate(prefix_reuse, questions=QUESTIONS, images=()):
    backend = PromptReadout(prefix_reuse)
    request = {"state": {"text": "hello"}, "questions": questions}
    if images:
        request["images"] = list(images)
    response, traces = DecisionEngine(backend).evaluate(request)
    return backend, response, traces


def test_prefix_and_suffix_concatenate_to_the_prompt():
    options = [("a", "x"), ("b", None)]
    for images in (0, 2):
        assert prefix_for({"s": 1}, images) + suffix_for("q", options) == prompt_for({"s": 1}, "q", options, images)
    assert prefix_for("s", 1) == f"<|im_start|>user\n{MEDIA_MARKER}State:\ns\n\n"
    assert suffix_for("q", options).startswith("Question: q\nOptions:\n[A] a: x\n")


@pytest.mark.parametrize("images", [(), (PNG_B64,)])
def test_grouped_answers_equal_separate_prompts(images):
    legacy, expected, _ = evaluate(False, images=images)
    grouped, actual, _ = evaluate(True, images=images)
    assert actual["answers"] == expected["answers"]
    assert actual["usage"]["input_tokens"] == expected["usage"]["input_tokens"]
    assert all(call[0] == "infer" for call in legacy.calls)
    # Round one: three questions and three chunks of the 120-option question; round two: its final.
    assert [call[0] for call in grouped.calls] == ["prefix", "prefix"]
    assert len(grouped.calls[0][2]) == 6 and len(grouped.calls[1][2]) == 1
    # Every prompt the grouped engine reads equals one the separate engine reads.
    assert ({call[1] + text for call in grouped.calls for text, _ in call[2]}
            == {call[1] for call in legacy.calls})
    assert all(call[3] == (list(images) or None) for call in grouped.calls)


def test_one_question_keeps_the_single_prompt_call():
    backend, _, _ = evaluate(True, {"yes": QUESTIONS["yes"]})
    assert [call[0] for call in backend.calls] == ["infer"]
    # A single chunked question still shares its prefix between chunks and the final round.
    backend, _, _ = evaluate(True, {"many": QUESTIONS["many"]})
    assert [call[0] for call in backend.calls] == ["prefix", "prefix"]


def test_image_tokens_count_once_and_shared_prefill_time_counts_once():
    _, response, _ = evaluate(True, {k: QUESTIONS[k] for k in ("route", "yes", "rating")}, images=(PNG_B64, PNG_B64))
    usage = response["usage"]
    assert usage["images"] == 2 and usage["image_tokens"] == 2048
    assert usage["prefill_ms"] == pytest.approx(3.0 + 1.0 + 3 * 0.5)


def test_generic_decisions_share_one_prefix_call():
    backend = PromptReadout()
    body = {"input": "ticket", "images": [PNG_B64], "questions": [
        {"id": "team", "type": "choice", "question": "team?", "options": [{"name": "a"}, {"name": "b"}]},
        {"id": "urgent", "type": "yes_no", "question": "urgent?"}]}
    with TestClient(create_app(DecisionEngine(backend))) as client:
        data = client.post("/v1/decisions", json=body).json()
    assert [call[0] for call in backend.calls] == ["prefix"]
    assert data["answers"]["team"]["image_tokens"] == 1024 == data["answers"]["urgent"]["image_tokens"]
    assert data["usage"]["prompt_tokens"] == sum(len(backend.calls[0][1] + text) for text, _ in backend.calls[0][2])


def test_wrong_result_count_is_a_backend_failure():
    class Short(PromptReadout):
        def infer_prefix(self, prefix, suffixes, images=None):
            response = super().infer_prefix(prefix, suffixes, images)
            response["results"].pop()
            return response
    with TestClient(create_app(DecisionEngine(Short()))) as client:
        response = client.post("/v1/systemone", json={"model": "shingi", "state": "s", "questions": {
            "a": QUESTIONS["yes"], "b": QUESTIONS["rating"]}})
    assert response.status_code == 503


def test_engine_option_and_older_readouts_disable_reuse():
    assert not DecisionEngine(PromptReadout(prefix_reuse=False)).prefix_reuse
    assert not DecisionEngine(PromptReadout(), prefix_reuse=False).prefix_reuse
    assert DecisionEngine(PromptReadout()).prefix_reuse


FAKE_READOUT = textwrap.dedent("""
    import json, sys
    print(json.dumps({"ready": True, "vision": False, "prefix_reuse": True, "prefix_cache_entries": 4,
                      "prefix_cache_bytes": 2147483648}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        results = [{"logits": [0.0] * len(s["labels"]), "input_tokens": 3} for s in request["suffixes"]]
        print(json.dumps({"results": results, "echo": line, "cache": {"entries": 1, "host_bytes": 123}}), flush=True)
""")


def test_native_prefix_request_line_and_cache_report(tmp_path, monkeypatch):
    script = tmp_path / "readout"
    script.write_text(f"#!{sys.executable}\n{FAKE_READOUT}")
    script.chmod(0o755)
    monkeypatch.setattr(backend_module, "gpu_profile", lambda: (None, 0, 0))
    monkeypatch.setattr(backend_module, "gpu_free_mib", lambda: 1 << 20)
    readout = backend_module.NativeReadout(script, "model.gguf")
    try:
        assert readout.prefix_reuse
        assert readout.prefix_cache == {"max_entries": 4, "max_bytes": 2147483648, "entries": 0, "host_bytes": 0}
        result = readout.infer_prefix("p", [("s1", ["A", "B"]), ("s2", ["A"])])
        assert json.loads(result["echo"]) == {"prefix": "p", "suffixes": [{"text": "s1", "labels": ["A", "B"]},
                                                                         {"text": "s2", "labels": ["A"]}]}
        uncached = json.loads(readout.infer_prefix("p", [("s", ["A"])], [PNG_B64], cache=False)["echo"])
        assert uncached["images"] == [PNG_B64] and uncached["cache"] is False
        assert readout.prefix_cache["entries"] == 1 and readout.prefix_cache["host_bytes"] == 123
        with TestClient(create_app(DecisionEngine(readout))) as client:
            version = client.get("/v1/version").json()
        assert version["prefix_reuse"] is True and version["prefix_cache"]["max_entries"] == 4
    finally:
        readout.close()
