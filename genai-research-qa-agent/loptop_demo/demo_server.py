"""Local FastAPI demo wrapping main.py's RAGPipeline.

Run:
    uv run uvicorn demo_server:app --reload --port 8080
Then open http://localhost:8080/
"""
import importlib.util
import json as jsonlib
import logging
import queue
import re
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HW2_DIR = Path(__file__).resolve().parent
STATIC_DIR = HW2_DIR / "demo_static"

spec = importlib.util.spec_from_file_location("main", HW2_DIR / "main.py")
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

logger = logging.getLogger("demo")
logger.setLevel(logging.DEBUG)
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.WARNING)
    logger.addHandler(h)

app = FastAPI(title="RAG Demo")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

pipeline: "v4.RAGPipeline | None" = None


@app.on_event("startup")
def _load_pipeline():
    global pipeline
    print("[startup] Loading embed + reranker (~30s first time, cached afterwards)...", flush=True)
    t0 = time.time()
    api_key = v4.LLM_API_KEY
    if v4.LLM_PROVIDER == "openrouter" and not api_key:
        print("[startup] WARNING: OPENROUTER_API_KEY not set, /query will fail at LLM step", flush=True)
    pipeline = v4.RAGPipeline(api_key=api_key, logger=logger)
    print(f"[startup] Ready in {time.time() - t0:.1f}s", flush=True)


class QueryReq(BaseModel):
    paper_text: str = Field(..., min_length=50)
    question: str = Field(..., min_length=3)
    title: str = Field("Demo paper", max_length=300)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record):
        try:
            self.records.append(record.getMessage())
        except Exception:
            pass


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "ready": pipeline is not None,
        "llm_model": v4.LLM_MODEL,
        "llm_provider": v4.LLM_PROVIDER,
        "embed_model": v4.EMBED_MODEL,
        "reranker_model": v4.RERANKER_MODEL,
    }


_TAG_MAP = {
    "Chunking": "chunk",
    "Indexing": "index",
    "QueryRoute": "route",
    "Query variants": "variants",
    "Query": "variants",
    "HyDE": "hyde",
    "Retrieval": "retrieval",
    "Rerank": "rerank",
    "RefineEvidence": "refine",
    "DynamicK": "dynk",
    "DynK": "dynk",
    "LLM": "llm",
    "LLM-fallback": "llm",
    "SC": "llm",
    "AntiEcho": "llm",
    "Recovery": "recovery",
    "EmptyFallback": "fallback",
    "Result": "result",
    "Time": "time",
}
_BRACKET_RE = re.compile(r"^\s*\[([^\]]+)\]")
_SKIP_SUBSTRINGS = ("[DynK] Stop:", "[SectionBoost]")


def _classify_log(msg: str):
    s = msg.rstrip()
    if not s.strip() or s.strip().startswith("==="):
        return None
    for bad in _SKIP_SUBSTRINGS:
        if bad in s:
            return None
    stripped = s.lstrip()
    m = _BRACKET_RE.match(stripped)
    if m:
        return _TAG_MAP.get(m.group(1).strip(), "info")
    if stripped.startswith(("Paper #", "Paper:", "Question:", "Text length:")):
        return "header"
    if s.startswith("  "):
        if stripped.startswith("Q") and ":" in stripped:
            return "variants"
        if stripped.startswith("#"):
            return "rerank"
        if stripped.startswith("EV["):
            return "refine"
        if ":" in stripped and "chunks" in stripped:
            return "chunk"
        return "detail"
    return None


@app.post("/query/stream")
def query_stream(req: QueryReq):
    if pipeline is None:
        raise HTTPException(503, "Pipeline still loading, retry in a few seconds.")

    entry = {"title": req.title, "full_text": req.paper_text, "question": req.question}
    events: queue.Queue = queue.Queue()
    result_box: dict = {}

    class _QueueHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = record.getMessage()
                tag = _classify_log(msg)
                if tag:
                    events.put(("log", {"tag": tag, "msg": msg.strip()}))
            except Exception:
                pass

    handler = _QueueHandler(level=logging.DEBUG)

    def worker():
        pipeline.logger.addHandler(handler)
        t0 = time.time()
        try:
            result = pipeline.process_paper(entry, 1, 1)
            result_box["result"] = result
            result_box["total_sec"] = round(time.time() - t0, 2)
        except Exception as e:
            result_box["error"] = str(e)
        finally:
            pipeline.logger.removeHandler(handler)
            events.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        yield f"data: {jsonlib.dumps({'type': 'start'})}\n\n"
        while True:
            kind, payload = events.get()
            if kind == "done":
                break
            yield f"data: {jsonlib.dumps({'type': 'log', **payload})}\n\n"
        if "error" in result_box:
            yield f"data: {jsonlib.dumps({'type': 'error', 'msg': result_box['error']})}\n\n"
            return
        r = result_box["result"]
        final = {
            "type": "result",
            "title": r.title,
            "answer": r.answer,
            "evidence": r.evidence,
            "total_sec": result_box["total_sec"],
            "config": {
                "embed": v4.EMBED_MODEL,
                "reranker": v4.RERANKER_MODEL,
                "llm": v4.LLM_MODEL,
                "provider": v4.LLM_PROVIDER,
            },
        }
        yield f"data: {jsonlib.dumps(final)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query")
def query(req: QueryReq):
    if pipeline is None:
        raise HTTPException(503, "Pipeline still loading, retry in a few seconds.")
    entry = {"title": req.title, "full_text": req.paper_text, "question": req.question}
    capture = _ListHandler()
    pipeline.logger.addHandler(capture)
    t0 = time.time()
    try:
        result = pipeline.process_paper(entry, 1, 1)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")
    finally:
        pipeline.logger.removeHandler(capture)

    return {
        "title": result.title,
        "answer": result.answer,
        "evidence": result.evidence,
        "trace": {
            "total_sec": round(time.time() - t0, 2),
            "log": capture.records,
            "config": {
                "embed": v4.EMBED_MODEL,
                "reranker": v4.RERANKER_MODEL,
                "llm": v4.LLM_MODEL,
                "provider": v4.LLM_PROVIDER,
            },
        },
    }
