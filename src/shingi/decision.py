"""Typed readout math; independent of the inference transport."""
from __future__ import annotations

import json
import math
import string
from dataclasses import dataclass
from typing import Protocol

LETTERS = string.ascii_uppercase + string.ascii_lowercase
MODEL_ID = "shingi-27b"
# mtmd's default media marker; the readout replaces each one with an image's tokens.
MEDIA_MARKER = "<__media__>"


def describe(value):
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def prompt_for(state, instructions, options, images=0):
    # Images open the user turn, in order, directly followed by the text: llama-server renders an
    # image part followed by a text part the same way for this chat template.
    # Verified against Prism /apply-template with enable_thinking=False.
    return prefix_for(state, images) + suffix_for(instructions, options)


def prefix_for(state, images=0):
    """The part of every prompt that depends only on the request: chat header, images and state.

    It ends after the blank line that follows the state, where the tokenizer also splits the full
    prompt; the native readout checks that split for each request before it reuses the prefix."""
    return f"<|im_start|>user\n{MEDIA_MARKER * images}State:\n{describe(state)}\n\n"


def suffix_for(instructions, options):
    """The per-question remainder of the prompt: question, lettered options and the assistant turn."""
    lines = "\n".join(f"[{LETTERS[i]}] {key}: {describe(value)}" for i, (key, value) in enumerate(options))
    return (f"Question: {describe(instructions)}\nOptions:\n{lines}\n\nAnswer with the letter of the best option only."
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def softmax(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("all candidate scores must be finite")
    peak = max(values)
    mass = [math.exp(x - peak) for x in values]
    return [x / sum(mass) for x in mass]


def choice_confidence(p):
    return 1.0 if len(p) == 1 else max(0.0, (len(p) * max(p) - 1) / (len(p) - 1))


def score_confidence(p):
    mode = max(range(len(p)), key=p.__getitem__)
    deviation = sum(value * abs(i - mode) for i, value in enumerate(p))
    uniform = sum(abs(i - (len(p) - 1) / 2) for i in range(len(p))) / len(p)
    return max(0.0, 1 - deviation / uniform)


@dataclass(frozen=True)
class Calibration:
    temperature: float = 1.0
    noul_temperature: float = 1.0
    noul_bias: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.temperature, self.noul_temperature, self.noul_bias)):
            raise ValueError("calibration must be finite")
        if self.temperature <= 0 or self.noul_temperature <= 0:
            raise ValueError("temperatures must be positive")


class Readout(Protocol):
    # Text-only calls pass no images argument; images are base64 strings in prompt order.
    def infer(self, prompt: str, labels: list[str], images: list[str] | None = None) -> dict: ...

    # Optional. A readout that also reports prefix_reuse evaluates the shared prefix once and returns
    # {"results": [one infer-shaped result per suffix], ...shared prefix timing and token counts}.
    # suffixes are (text, labels) pairs.
    # def infer_prefix(self, prefix: str, suffixes: list[tuple[str, list[str]]],
    #                  images: list[str] | None = None) -> dict: ...


def chunked(options):
    """Prompts with more options than letters are split into chunks, then a final round over the chunk winners."""
    if len(options) <= len(LETTERS):
        return None
    n_chunks = math.ceil(len(options) / len(LETTERS))
    chunk_size = math.ceil(len(options) / n_chunks)
    return [options[i:i + chunk_size] for i in range(0, len(options), chunk_size)]


class DecisionEngine:
    def __init__(self, backend: Readout, calibration=Calibration(), *, model_id=MODEL_ID, prefix_reuse=True):
        self.backend = backend
        self.calibration = calibration
        self.model_id = model_id
        # Evaluate the shared prefix (header, images and state) once per request when the readout supports it.
        self.prefix_reuse = prefix_reuse and bool(getattr(backend, "prefix_reuse", False))

    @property
    def vision(self):
        return bool(getattr(self.backend, "vision", False))

    def _infer_all(self, state, images, prompts, single=False):
        """Read out every (instructions, options) prompt over the same state and images, in order."""
        labels = [list(LETTERS[:len(options)]) for _, options in prompts]
        # A request with exactly one prompt keeps the original single-prompt call: there is nothing to share.
        if self.prefix_reuse and not (single and len(prompts) == 1):
            response = self.backend.infer_prefix(prefix_for(state, len(images)),
                                                 [(suffix_for(i, o), l) for (i, o), l in zip(prompts, labels)],
                                                 list(images) or None)
            results = response["results"]
            if len(results) != len(prompts):
                raise RuntimeError("the readout returned the wrong number of prefix results")
            shared = {key: response[key] for key in ("images", "image_tokens") if key in response}
            results = [{**result, **shared} for result in results]
            # The shared prefix is evaluated (or restored from the readout's cache) once per call.
            results[0]["shared_prefill_ms"] = response.get("prefix_ms", 0) + response.get("snapshot_ms", 0)
            return results
        return [self.backend.infer(prompt_for(state, i, o, len(images)), l, list(images)) if images
                else self.backend.infer(prompt_for(state, i, o), l)
                for (i, o), l in zip(prompts, labels)]

    def _p(self, result, n, temperature):
        logits = result["logits"]
        if len(logits) != n or any(not math.isfinite(x) for x in logits):
            raise RuntimeError("incomplete or non-finite candidate logits")
        return softmax([x / self.calibration.temperature / temperature for x in logits])

    def distributions(self, state, jobs, images=(), temperature=1.0):
        """[(instructions, options)] -> [(p, traces)]. All prompts of one round share one prefix call."""
        prompts, first = [], []
        for instructions, options in jobs:
            parts = chunked(options) or [options]
            first.append(len(prompts))
            prompts += [(instructions, part) for part in parts]
        results = self._infer_all(state, images, prompts, single=True)
        out, finals = [], []
        for (instructions, options), start in zip(jobs, first):
            chunks = chunked(options)
            if chunks is None:
                result = results[start]
                out.append((self._p(result, len(options), temperature), [result]))
                continue
            # OpenJev-style approximate anchor composition. All labels retain mass;
            # this is not a single forward pass or an exact global softmax.
            parts = [(self._p(r, len(c), temperature), [r]) for r, c in zip(results[start:start + len(chunks)], chunks)]
            winners = [max(range(len(p)), key=p.__getitem__) for p, _ in parts]
            finals.append((len(out), instructions, [c[w] for c, w in zip(chunks, winners)], parts, winners))
            out.append(None)
        if finals:
            final_results = self._infer_all(state, images, [(i, o) for _, i, o, _, _ in finals])
            for (index, _, final_options, parts, winners), result in zip(finals, final_results):
                final, traces = self._p(result, len(final_options), temperature), [result]
                values = [final[j] * value / p[w] for j, ((p, _), w) in enumerate(zip(parts, winners)) for value in p]
                total = sum(values)
                out[index] = ([x / total for x in values], [trace for _, batch in parts for trace in batch] + traces)
        return out

    def distribution(self, state, instructions, options, images=(), temperature=1.0):
        return self.distributions(state, [(instructions, options)], images, temperature)[0]

    @staticmethod
    def _options(q):
        instructions = q["instructions"]
        kind = q["type"]
        if kind == "choice":
            # Canonical order: reordering the same JSON map gives an identical prompt.
            options = sorted(q["criteria"].items(), key=lambda item: item[0])
        elif kind == "score":
            options = [(str(i), value) for i, value in enumerate(q["criteria"])]
            instructions = describe(instructions) + " Rate along the ordered levels below (lowest first)."
        else:
            criteria = q.get("criteria") or {}
            options = [("yes", criteria.get("true") or "The statement is true."),
                       ("no", criteria.get("false") or "The statement is false.")]
        return kind, instructions, options

    def _finish(self, kind, options, p, traces, temperature):
        if kind == "noul":
            # Work from logits, not clipped probabilities, to retain the tails.
            logits = traces[0]["logits"]
            log_odds = ((logits[0] - logits[1]) / self.calibration.temperature / self.calibration.noul_temperature
                        + self.calibration.noul_bias) / temperature
            yes = softmax([log_odds, 0.0])[0]
            return {"type": kind, "noul": yes}
        probabilities = {key: value for (key, _), value in zip(options, p)}
        if kind == "choice":
            winner = max(range(len(p)), key=p.__getitem__)
            return {"type": kind, "choice": options[winner][0], "probabilities": probabilities,
                    "confidence": choice_confidence(p)}
        return {"type": kind, "score": sum(i * value for i, value in enumerate(p)),
                "legend": dict(options), "probabilities": probabilities, "confidence": score_confidence(p)}

    def answer_all(self, state, questions, images=(), temperature=1.0):
        """Answer several questions about one state and image set: [(answer, traces)] in order."""
        parsed = [self._options(q) for q in questions]
        distributions = self.distributions(state, [(i, o) for _, i, o in parsed], images, temperature)
        return [(self._finish(kind, options, p, traces, temperature), traces)
                for (kind, _, options), (p, traces) in zip(parsed, distributions)]

    def answer(self, state, q, images=(), temperature=1.0):
        return self.answer_all(state, [q], images, temperature)[0]

    def evaluate(self, request, temperature=1.0):
        images = request.get("images") or []
        keys = list(request["questions"])
        results = self.answer_all(request["state"], [request["questions"][k] for k in keys], images, temperature)
        answers = {k: answer for k, (answer, _) in zip(keys, results)}
        traces = {k: trace for k, (_, trace) in zip(keys, results)}
        calls = [x for group in traces.values() for x in group]
        usage = {"input_tokens": sum(x["input_tokens"] for x in calls), "output_tokens": 0}
        if images:
            # The images are encoded once per request (or restored with the shared prefix), so their
            # tokens count once; input_tokens still counts every prompt in full.
            usage |= {"images": len(images), "image_tokens": max(x.get("image_tokens", 0) for x in calls),
                      "prefill_ms": round(sum(x.get("prefill_ms", 0) + x.get("shared_prefill_ms", 0)
                                              for x in calls), 3)}
        return {"model": self.model_id, "answers": answers, "usage": usage}, traces


def label_mass(trace):
    """Full-vocabulary probability of the candidate labels, when the readout reports its normalizer."""
    normalizer = trace.get("log_normalizer")
    if normalizer is None:
        return None
    return math.fsum(math.exp(x - normalizer) for x in trace["logits"])
