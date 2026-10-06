import hashlib
import json
from pathlib import Path

import pytest

from shingi import model, server
from shingi.decision import Calibration

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_calibration_matches_the_pinned_release():
    calibration, digest = model.load_calibration(ROOT / "calibration.json")
    assert digest == model.CALIBRATION_SHA256
    assert calibration == Calibration(temperature=1.0, noul_temperature=1.0, noul_bias=0.0)
    provenance = json.loads((ROOT / "calibration.json").read_text())["provenance"]
    assert provenance["model_sha256"] == model.MODEL_SHA256 and provenance["adapter_sha256"] is None


def test_fetch_uses_the_release_repo_and_revision(monkeypatch):
    calls = []
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, filename, revision: calls.append((repo, filename, revision)) or f"/cache/{filename}")
    monkeypatch.delenv("SHINGI_REVISION", raising=False)
    assert model.fetch("calibration.json") == Path("/cache/calibration.json")
    monkeypatch.setenv("SHINGI_REVISION", "abc123")
    model.fetch("shingi-27b.gguf")
    assert calls == [("kortexa-ai/shingi-27b", "calibration.json", model.HF_REVISION),
                     ("kortexa-ai/shingi-27b", "shingi-27b.gguf", "abc123")]


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    weights = tmp_path / "shingi-27b.gguf"
    weights.write_bytes(b"not really a model")
    monkeypatch.setattr(server, "MODEL_SHA256", hashlib.sha256(weights.read_bytes()).hexdigest())
    fetched = []
    monkeypatch.setattr(server, "fetch", lambda name: fetched.append(name) or
                        {"calibration.json": ROOT / "calibration.json", "shingi-27b.gguf": weights}[name])
    return weights, fetched


def test_prepare_downloads_omitted_files_and_verifies_them(fake_release):
    weights, fetched = fake_release
    path, identity, calibration, calibration_sha256 = server.prepare()
    assert fetched == ["calibration.json", "shingi-27b.gguf"]
    assert path == weights
    assert identity == {"model": "shingi-27b", "model_sha256": server.MODEL_SHA256, "verified": True}
    assert calibration.noul_temperature == 1.0 and calibration_sha256 == model.CALIBRATION_SHA256


def test_prepare_rejects_wrong_weights_unless_skipped(fake_release, tmp_path):
    other = tmp_path / "other.gguf"
    other.write_bytes(b"different")
    with pytest.raises(RuntimeError, match="Model SHA-256"):
        server.prepare(model=other)
    path, identity, _, _ = server.prepare(model=other, skip_verify=True)
    assert path == other and identity["verified"] is False and identity["model_sha256"] is None


def test_prepare_rejects_a_different_calibration_unless_skipped(fake_release, tmp_path):
    custom = tmp_path / "calibration.json"
    custom.write_text(json.dumps({"parameters": {"temperature": 2.0}}))
    with pytest.raises(RuntimeError, match="Calibration SHA-256"):
        server.prepare(calibration=custom)
    _, _, calibration, digest = server.prepare(calibration=custom, skip_verify=True)
    assert calibration.temperature == 2.0 and digest == hashlib.sha256(custom.read_bytes()).hexdigest()
