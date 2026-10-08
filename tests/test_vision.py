"""Image input and the SGLang /v1/decisions route, with fake readouts; no GPU or model needed."""
import base64
import hashlib
import json
import math
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from shingi import backend as backend_module
from shingi import server
from shingi.decision import MEDIA_MARKER, DecisionEngine, prompt_for
from shingi.model import PROJECTOR
from shingi.schema import InputDecisionRequest, Request, image_base64
from shingi.server import create_app

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
PNG_B64 = base64.b64encode(PNG).decode()
JPEG_B64 = base64.b64encode(JPEG).decode()


class VisionReadout:
    """Records calls; text-only calls use the original two-argument signature."""

    def __init__(self, probabilities=None, vision=True):
        self.probabilities = probabilities
        self.vision = vision
        self.calls = []

    def infer(self, prompt, labels, *images):
        self.calls.append((prompt, labels, *images))
        p = self.probabilities or [1 / len(labels)] * len(labels)
        result = {"logits": [math.log(x) for x in p], "input_tokens": 50, "prefill_ms": 2.0,
                  # Labels hold half of the full-vocabulary mass.
                  "log_normalizer": math.log(2.0)}
        if images:
            result |= {"images": len(images[0]), "image_tokens": 1024 * len(images[0])}
            result["input_tokens"] += result["image_tokens"]
        return result


def client(backend):
    return TestClient(create_app(DecisionEngine(backend)))


def systemone(question, **extra):
    return {"model": "shingi-27b", "state": "a picture", "questions": {"q": question}, **extra}


# Image validation

@pytest.mark.parametrize("value", [PNG_B64, f"data:image/png;base64,{PNG_B64}", JPEG_B64,
                                   base64.b64encode(b"GIF89a" + b"\x00" * 8).decode(),
                                   base64.b64encode(b"BM" + b"\x00" * 8).decode()])
def test_supported_images_normalize_to_plain_base64(value):
    decoded = base64.b64decode(image_base64(value), validate=True)
    assert decoded == base64.b64decode(value.partition(",")[2] or value)


@pytest.mark.parametrize("value, message", [
    ("not base64!", "base64"),
    ("", "empty"),
    ("data:text/plain;base64," + PNG_B64, "data URL"),
    ("data:image/png," + PNG_B64, "data URL"),
    (base64.b64encode(b"RIFF\x00\x00\x00\x00WEBPVP8 ").decode(), "unsupported image format"),
    (base64.b64encode(b"plain text").decode(), "unsupported image format"),
])
def test_bad_images_are_input_errors(value, message):
    with pytest.raises(ValueError, match=message):
        image_base64(value)


def test_at_most_eight_images_and_no_marker_text():
    question = {"type": "noul", "instructions": "yes?"}
    Request.model_validate(systemone(question, images=[PNG_B64] * 8))
    with pytest.raises(ValueError, match="at most 8"):
        Request.model_validate(systemone(question, images=[PNG_B64] * 9))
    with pytest.raises(ValueError, match="media marker"):
        Request.model_validate({**systemone(question, images=[PNG_B64]), "state": f"see {MEDIA_MARKER}"})
    with pytest.raises(ValueError, match="media marker"):
        Request.model_validate(systemone({"type": "choice", "instructions": "pick",
                                          "criteria": {"a": MEDIA_MARKER}}, images=[PNG_B64]))
    # Without images the marker is ordinary text.
    Request.model_validate({**systemone(question), "state": f"see {MEDIA_MARKER}"})


# Prompt and marker placement

def test_images_open_the_user_turn_in_order_before_the_text():
    text_only = prompt_for("s", "q", [("yes", "y"), ("no", "n")])
    with_images = prompt_for("s", "q", [("yes", "y"), ("no", "n")], images=2)
    assert with_images == text_only.replace("<|im_start|>user\n", f"<|im_start|>user\n{MEDIA_MARKER * 2}", 1)
    assert with_images.startswith(f"<|im_start|>user\n{MEDIA_MARKER}{MEDIA_MARKER}State:\ns\n")
    assert with_images.count(MEDIA_MARKER) == 2


def test_text_only_requests_use_the_unchanged_call_and_prompt():
    backend = VisionReadout([.8, .2])
    with client(backend) as c:
        response = c.post("/v1/systemone", json=systemone({"type": "noul", "instructions": "yes?"}))
    assert response.status_code == 200
    prompt, labels = backend.calls[0]
    assert MEDIA_MARKER not in prompt and labels == ["A", "B"]
    assert response.json()["usage"] == {"input_tokens": 50, "output_tokens": 0}


def test_systemone_images_reach_every_question_in_order():
    backend = VisionReadout([.8, .2])
    body = {"model": "shingi", "state": "x", "images": [PNG_B64, f"data:image/jpeg;base64,{JPEG_B64}"],
            "questions": {"a": {"type": "noul", "instructions": "yes?"},
                          "b": {"type": "choice", "instructions": "pick", "criteria": {"l": None, "r": None}}}}
    with client(backend) as c:
        response = c.post("/v1/systemone", json=body)
    assert response.status_code == 200, response.text
    assert [call[2] for call in backend.calls] == [[PNG_B64, JPEG_B64]] * 2
    assert all(call[0].count(MEDIA_MARKER) == 2 for call in backend.calls)
    usage = response.json()["usage"]
    # The images count once per request; every prompt still counts in full.
    assert usage["images"] == 2 and usage["image_tokens"] == 2048 and usage["input_tokens"] == 2 * (50 + 2048)


def test_images_without_a_projector_are_rejected_before_inference():
    backend = VisionReadout(vision=False)
    with client(backend) as c:
        response = c.post("/v1/systemone", json=systemone({"type": "noul", "instructions": "x"}, images=[PNG_B64]))
        version = c.get("/v1/version").json()
    assert response.status_code == 422 and "--no-vision" in response.text
    assert backend.calls == []
    assert version["image_input"] is False and version["max_images"] == 0 and version["projector"] is None


def test_version_reports_the_loaded_projector():
    with client(VisionReadout()) as c:
        version = c.get("/v1/version").json()
        description = c.get("/v1/models").json()["models"][0]["description"]
    assert version["image_input"] is True and version["max_images"] == 8
    assert version["projector"]["sha256"] == PROJECTOR["sha256"]
    assert "images" in description


# /v1/decisions, System One shaped body (SGLang decision-model route)

def test_state_shaped_decisions_add_decision_and_yes_no_probabilities():
    backend = VisionReadout([.75, .25])
    body = {"state": {"task": "inspect"}, "images": [f"data:image/png;base64,{PNG_B64}"],
            "questions": {"red": {"type": "noul", "instructions": "Is it red?"},
                          "side": {"type": "choice", "instructions": "Which side?",
                                   "criteria": {"left": "The left half", "right": "The right half"}},
                          "count": {"type": "score", "instructions": "How many?", "criteria": ["0", "1", "2", "3"]}}}
    backend.probabilities = None
    with client(backend) as c:
        response = c.post("/v1/decisions", json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    red = data["answers"]["red"]
    assert red["probabilities"]["yes"] == pytest.approx(red["noul"])
    assert red["decision"] in ("yes", "no") and red["confidence"] == pytest.approx(max(red["noul"], 1 - red["noul"]))
    # Uniform logits: ties go to the smaller label.
    assert data["answers"]["side"]["decision"] == "left" == data["answers"]["side"]["choice"]
    assert data["answers"]["count"]["decision"] == "0" and data["answers"]["count"]["legend"]["3"] == "3"
    assert data["usage"]["decision_count"] == 3 and data["usage"]["images"] == 1
    assert "calibration" not in data


def test_state_shaped_temperature_divides_logits_and_is_echoed():
    backend = VisionReadout([.8, .2])
    body = {"state": "s", "temperature": 2.0,
            "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}}}}
    with client(backend) as c:
        data = c.post("/v1/decisions", json=body).json()
    assert data["answers"]["q"]["probabilities"]["a"] == pytest.approx(2 / 3)
    assert data["calibration"] == {"method": "temperature-scaling", "temperature": 2.0}


@pytest.mark.parametrize("body", [
    {"state": "s", "questions": {str(i): {"type": "noul", "instructions": "x"} for i in range(17)}},
    {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "thinking": {"enabled": True}},
    {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "temperature": 0},
    {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "images": ["abc"]},
])
def test_state_shaped_invalid_bodies_fail_before_inference(body):
    backend = VisionReadout()
    with client(backend) as c:
        assert c.post("/v1/decisions", json=body).status_code == 422
    assert backend.calls == []


# /v1/decisions, generic body (SGLang main)

GENERIC = {
    "input": "I've been trying to connect my Stripe account for 3 days and the integration keeps failing.",
    "questions": [
        {"id": "team", "type": "choice", "question": "Which team should handle this ticket?",
         "options": [{"name": "technical", "description": "Bugs or integration problems"},
                     {"name": "billing", "description": "Payment or subscription issues"},
                     {"name": "sales"}]},
        {"id": "frustration", "type": "score", "question": "How frustrated is the customer?",
         "levels": ["Calm", "Frustrated but civil", "Very angry"]},
        {"id": "urgent", "type": "yes_no", "question": "The customer needs an answer today.", "yes": "Today"},
    ],
}


def test_generic_decisions_map_onto_the_engine():
    backend = VisionReadout()
    with client(backend) as c:
        response = c.post("/v1/decisions", json=GENERIC)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["object"] == "decisions" and data["model"] == "shingi-27b" and data["prompt_format_version"] == 1
    team, frustration, urgent = (data["answers"][k] for k in ("team", "frustration", "urgent"))
    assert team["type"] == "choice" and set(team["probabilities"]) == {"technical", "billing", "sales"}
    assert team["choice"] == "billing"  # uniform: the canonical (sorted) first option
    assert frustration["type"] == "score" and set(frustration["probabilities"]) == {"0", "1", "2"}
    assert frustration["score"] == pytest.approx(1)
    assert urgent == {"type": "yes_no", "probabilities": {"yes": .5, "no": .5}, "label_mass": .5}
    assert team["label_mass"] == pytest.approx(.5)
    # Descriptions reach the prompt; the yes description becomes the yes option, the no default stays.
    prompts = [call[0] for call in backend.calls]
    assert "[C] technical: Bugs or integration problems" in prompts[0]
    assert "[A] yes: Today" in prompts[2] and "[B] no: The statement is false." in prompts[2]
    assert data["usage"] == {"prompt_tokens": 150, "completion_tokens": 0, "total_tokens": 150}


def test_generic_decisions_accept_images():
    backend = VisionReadout()
    body = {**GENERIC, "images": [PNG_B64]}
    with client(backend) as c:
        data = c.post("/v1/decisions", json=body).json()
    assert all(call[2] == [PNG_B64] for call in backend.calls)
    assert data["answers"]["urgent"]["image_tokens"] == 1024


def replace_question(**changes):
    return {**GENERIC, "questions": [{**GENERIC["questions"][0], **changes}]}


@pytest.mark.parametrize("body", [
    {**GENERIC, "input": "  "},
    {**GENERIC, "questions": []},
    {**GENERIC, "questions": [GENERIC["questions"][2], GENERIC["questions"][2]]},
    replace_question(options=[{"name": "a"}]),
    replace_question(options=[{"name": str(i)} for i in range(27)]),
    replace_question(options=[{"name": "Same"}, {"name": " same "}]),
    replace_question(options=[{"name": "a\nb"}, {"name": "c"}]),
    replace_question(type="bogus"),
    {**GENERIC, "questions": [{"id": "s", "type": "score", "question": "q", "levels": [str(i) for i in range(11)]}]},
    {**GENERIC, "prompt_format_version": 2},
    {**GENERIC, "return_prompt_token_ids": True},
    {**GENERIC, "chat_template_kwargs": {"enable_thinking": True}},
    {**GENERIC, "unknown": 1},
    {**GENERIC, "images": [PNG_B64] * 9},
])
def test_generic_invalid_bodies_fail_before_inference(body):
    backend = VisionReadout()
    with client(backend) as c:
        assert c.post("/v1/decisions", json=body).status_code == 422
    assert backend.calls == []


def test_generic_body_accepts_disabled_thinking_and_the_served_format():
    InputDecisionRequest.model_validate({**GENERIC, "chat_template_kwargs": {"enable_thinking": False},
                                         "prompt_format_version": 1, "model": "anything"})


# Native process wiring

FAKE_READOUT = textwrap.dedent("""
    import json, sys
    args = sys.argv[1:-2] if sys.argv[-2:-1] == ["--parallel"] else sys.argv[1:]
    vision = len(args) == 3
    print(json.dumps({"ready": True, "vision": vision, "argv": args}), flush=True)
    for line in sys.stdin:
        print(json.dumps({"logits": [0.0, 0.0], "input_tokens": 1, "echo": line}), flush=True)
""")


@pytest.fixture
def fake_native(tmp_path, monkeypatch):
    script = tmp_path / "readout"
    script.write_text(f"#!{sys.executable}\n{FAKE_READOUT}")
    script.chmod(0o755)
    monkeypatch.setattr(backend_module, "gpu_profile", lambda: (None, 0, 0))
    monkeypatch.setattr(backend_module, "gpu_free_mib", lambda: 1 << 20)
    return script


def test_text_request_line_is_unchanged_and_images_are_added_only_when_sent(fake_native):
    readout = backend_module.NativeReadout(fake_native, "model.gguf", "mmproj.gguf")
    try:
        assert readout.vision and readout.info["argv"] == ["model.gguf", "16384", "mmproj.gguf"]
        text = readout.infer("p", ["A", "B"])["echo"]
        assert text == json.dumps({"prompt": "p", "labels": ["A", "B"]}) + "\n"
        image = json.loads(readout.infer("p", ["A", "B"], [PNG_B64])["echo"])
        assert image == {"prompt": "p", "labels": ["A", "B"], "images": [PNG_B64]}
    finally:
        readout.close()
    text_only = backend_module.NativeReadout(fake_native, "model.gguf")
    try:
        assert not text_only.vision and text_only.info["argv"] == ["model.gguf", "16384"]
    finally:
        text_only.close()


def test_projector_is_fetched_and_verified(tmp_path, monkeypatch):
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"projector")
    monkeypatch.setattr(server, "fetch_projector", lambda: projector)
    monkeypatch.setitem(server.PROJECTOR, "sha256", hashlib.sha256(b"projector").hexdigest())
    assert server.prepare_projector() == (projector, True)
    other = tmp_path / "other.gguf"
    other.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="Vision projector SHA-256"):
        server.prepare_projector(other)
    assert server.prepare_projector(other, skip_verify=True) == (other, False)
