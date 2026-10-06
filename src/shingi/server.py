import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn

from .backend import CONTEXT_TOKENS, NativeReadout
from .decision import Calibration, DecisionEngine, MODEL_ID, label_mass
from .model import (CALIBRATION_FILE, CALIBRATION_SHA256, MODEL_FILE, MODEL_SHA256, PROJECTOR, RUNTIME,
                    fetch, fetch_projector, load_calibration, sha256, verify)
from .schema import (DecisionsRequest, InputDecisionRequest, MAX_IMAGES, PROMPT_FORMAT_VERSION, Request,
                     StateDecisionRequest)

ALIASES = (MODEL_ID, "shingi", "shingi-latest", "jev-latest", "openjev")


def state_answer(answer):
    """A System One answer in the SGLang decision-model shape: adds decision, and yes/no probabilities for noul."""
    answer = dict(answer)
    if answer["type"] == "noul":
        yes = answer["noul"]
        answer["probabilities"] = {"yes": yes, "no": 1 - yes}
        answer["decision"] = "yes" if yes >= .5 else "no"
        answer["confidence"] = max(yes, 1 - yes)
    else:
        p = answer["probabilities"]
        # Ties go to the smaller label.
        answer["decision"] = min(p, key=lambda key: (-p[key], key))
    return answer


def input_answers(body, engine):
    """Answer an SGLang generic /v1/decisions body with the System One engine."""
    images = body.images or []
    questions = []
    for q in body.questions:
        if q.type == "choice":
            questions.append({"type": "choice", "instructions": q.question,
                              "criteria": {o.name: o.description for o in q.options}})
        elif q.type == "score":
            questions.append({"type": "score", "instructions": q.question, "criteria": q.levels})
        else:
            questions.append({"type": "noul", "instructions": q.question,
                              "criteria": {k: v for k, v in (("true", q.yes), ("false", q.no)) if v is not None} or None})
    answers, prompt_tokens = {}, 0
    # All questions share one evaluation of the input and images.
    for q, (answer, traces) in zip(body.questions, engine.answer_all(body.input, questions, images, body.temperature)):
        prompt_tokens += sum(t["input_tokens"] for t in traces)
        masses = [label_mass(t) for t in traces]
        out = {"type": q.type, "label_mass": None if None in masses else masses[-1]}
        if q.type == "yes_no":
            out["probabilities"] = {"yes": answer["noul"], "no": 1 - answer["noul"]}
        elif q.type == "choice":
            out |= {"probabilities": answer["probabilities"], "choice": answer["choice"]}
        else:
            out |= {"probabilities": answer["probabilities"], "score": answer["score"]}
        if images:
            out["image_tokens"] = traces[-1].get("image_tokens")
        answers[q.id] = out
    return answers, prompt_tokens


def create_app(engine=None, *, executable=None, model=None, identity=None,
               calibration=Calibration(), calibration_sha256=None, projector=None):
    identity = identity or {"model": MODEL_ID, "model_sha256": None, "verified": False}

    @asynccontextmanager
    async def lifespan(app):
        backend = None
        try:
            if engine is None:
                backend = NativeReadout(executable, model, projector)
                app.state.engine = DecisionEngine(backend, calibration, model_id=identity["model"])
            yield
        finally:
            if backend is not None:
                backend.close()

    app = FastAPI(title="Shingi 27B", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Report where and why, but never echo the request body back (images can be megabytes).
        errors = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})
    app.state.engine = engine

    @app.get("/health")
    def health():
        current = app.state.engine
        process = getattr(getattr(current, "backend", None), "process", None)
        if current is None or (process is not None and process.poll() is not None):
            raise HTTPException(503, "native model unavailable")
        return {"status": "ok"}

    @app.get("/v1/version")
    def version():
        vision = app.state.engine.vision
        return {**identity, "calibration": asdict(app.state.engine.calibration),
                "calibration_sha256": calibration_sha256, "runtime": RUNTIME,
                "context_tokens": CONTEXT_TOKENS, "kv_cache": "q8_0", "choice_order": "sorted",
                "single_pass_options": 52, "choice_limit": 255, "image_input": vision,
                "max_images": MAX_IMAGES if vision else 0,
                "projector": PROJECTOR if vision else None,
                "prefix_reuse": app.state.engine.prefix_reuse,
                "prefix_cache": getattr(app.state.engine.backend, "prefix_cache", None)}

    @app.get("/v1/models")
    def models():
        kind = "text and images" if app.state.engine.vision else "text only"
        return {"models": [{"name": identity["model"], "description": f"Shingi 27B local decision model; {kind}.",
                            "release_date": "2026-09-29"}]}

    def run(images, work):
        if images and not app.state.engine.vision:
            raise HTTPException(422, "image input requires the vision projector; the server runs with --no-vision")
        try:
            return work()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, TimeoutError, BrokenPipeError) as exc:
            raise HTTPException(503, "native readout unavailable") from exc

    @app.post("/v1/systemone")
    def system_one(body: Request):
        if body.model not in ALIASES:
            raise HTTPException(422, "unknown model alias")
        return run(body.images, lambda: app.state.engine.evaluate(body.model_dump(exclude_none=True))[0])

    @app.post("/v1/decisions")
    def decisions(body: DecisionsRequest):
        # The SGLang decision route: a System One shaped body (state and keyed questions) or the
        # generic body (input and a list of typed questions). Any model name is answered by Shingi.
        engine = app.state.engine
        if isinstance(body, StateDecisionRequest):
            request = body.model_dump(exclude_none=True, exclude={"temperature", "thinking", "model"})

            def work():
                response, _ = engine.evaluate(request, body.temperature or 1.0)
                usage = response["usage"] | {"decision_count": len(response["answers"])}
                result = {"model": response["model"], "answers": {k: state_answer(v) for k, v in response["answers"].items()},
                          "usage": usage}
                if body.temperature is not None:
                    result["calibration"] = {"method": "temperature-scaling", "temperature": body.temperature}
                return result
            return run(body.images, work)
        assert isinstance(body, InputDecisionRequest)

        def work():
            answers, prompt_tokens = input_answers(body, engine)
            return {"object": "decisions", "model": engine.model_id, "prompt_format_version": PROMPT_FORMAT_VERSION,
                    "answers": answers,
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens}}
        return run(body.images, work)

    return app


def prepare_projector(projector=None, skip_verify=False):
    """Fetch the vision projector when omitted, verify its pinned hash, and return (path, verified)."""
    projector = projector or fetch_projector()
    if skip_verify:
        return projector, False
    print(f"Verifying {projector} ...", flush=True)
    verify("Vision projector", sha256(projector), PROJECTOR["sha256"])
    return projector, True


def prepare(model=None, calibration=None, skip_verify=False):
    """Fetch omitted files from the Hub, verify pinned hashes, and return the serving identity."""
    calibration_path = calibration or fetch(CALIBRATION_FILE)
    parameters, calibration_sha256 = load_calibration(calibration_path)
    if not skip_verify:
        verify("Calibration", calibration_sha256, CALIBRATION_SHA256)
    model = model or fetch(MODEL_FILE)
    model_sha256 = None
    if not skip_verify:
        print(f"Verifying {model} ...", flush=True)
        model_sha256 = sha256(model)
        verify("Model", model_sha256, MODEL_SHA256)
    identity = {"model": MODEL_ID, "model_sha256": model_sha256, "verified": not skip_verify}
    return model, identity, parameters, calibration_sha256


def main():
    parser = argparse.ArgumentParser(prog="shingi-27b", description="Serve the Shingi 27B decision API.")
    parser.add_argument("--executable", required=True, type=Path,
                        help="native readout built against the pinned Prism runtime")
    parser.add_argument("--model", type=Path, help=f"{MODEL_FILE}; downloaded from Hugging Face when omitted")
    parser.add_argument("--calibration", type=Path,
                        help=f"{CALIBRATION_FILE}; downloaded from Hugging Face when omitted")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mmproj", type=Path,
                        help=f"vision projector ({PROJECTOR['filename']}); downloaded from Hugging Face when omitted")
    parser.add_argument("--no-vision", action="store_true",
                        help="do not load the vision projector: text only, less memory")
    parser.add_argument("--skip-verify", action="store_true",
                        help="do not check the pinned SHA-256 of the model, calibration and projector")
    args = parser.parse_args()
    if not args.executable.is_file():
        parser.error(f"readout executable not found: {args.executable}")
    if args.no_vision and args.mmproj:
        parser.error("--mmproj and --no-vision are mutually exclusive")
    model, identity, calibration, calibration_sha256 = prepare(args.model, args.calibration, args.skip_verify)
    projector = None
    if not args.no_vision:
        projector, identity["projector_verified"] = prepare_projector(args.mmproj, args.skip_verify)
    # The API has no authentication. Bind beyond localhost only behind your own access control.
    uvicorn.run(create_app(executable=args.executable, model=model, identity=identity,
                           calibration=calibration, calibration_sha256=calibration_sha256, projector=projector),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
