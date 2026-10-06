"""Pinned identity of the released model, its calibration and its runtime."""
import hashlib
import json
import os
from pathlib import Path

from .decision import Calibration

HF_REPO = "kortexa-ai/shingi-27b"
# Hugging Face commit that holds exactly the weights and calibration pinned below.
HF_REVISION = "d02406fc8974a91ebf45b0e7d46c5381ee42e1c2"
MODEL_FILE = "shingi-27b.gguf"
MODEL_SHA256 = "c62ae5b61e458aa70aa2f51ebbd193d50e00d170cb318c60341bb68c6471242c"
CALIBRATION_FILE = "calibration.json"
CALIBRATION_SHA256 = "bd572be4ef9a72902757f420ea2f5742e73c2627e3f372bf277baba1306f76db"
RUNTIME = {"repository": "https://github.com/PrismML-Eng/llama.cpp",
           "revision": "d8f26eec76da6d09bb708bcba51ef64b8cd868a3"}
# Bonsai 2 27B vision projector (Apache-2.0), used unchanged. One pinned source.
PROJECTOR = {"repository": "kortexa-ai/shingi-27b",
             "revision": "262cde012ae9e57d2d37e6040e810ca8713ccc40",
             "filename": "mmproj.gguf",
             "sha256": "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(filename):
    """Download a release file into the standard Hugging Face cache (honours HF_HOME and HF_TOKEN)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, filename, revision=os.environ.get("SHINGI_REVISION", HF_REVISION)))


def fetch_projector():
    """Download the pinned vision projector into the standard Hugging Face cache."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(PROJECTOR["repository"], PROJECTOR["filename"], revision=PROJECTOR["revision"]))


def verify(what, actual, expected):
    if actual != expected:
        raise RuntimeError(f"{what} SHA-256 is {actual}, expected {expected}. "
                           "Delete the file to download it again, or pass --skip-verify to use it anyway.")


def load_calibration(path):
    data = Path(path).read_bytes()
    return Calibration(**json.loads(data)["parameters"]), hashlib.sha256(data).hexdigest()
