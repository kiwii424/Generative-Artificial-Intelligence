import os
import re
import math
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CHILD_TARGET_CHARS = 220
CHILD_MAX_CHARS = 420
SENTENCE_OVERLAP = 1
BM25_TOP_K = 20
FINAL_K = 3

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="GenAI Research QA Agent")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    paper_text: str = Field(..., min_length=50)
    question: str = Field(..., min_length=3)
    top_k: int = Field(FINAL_K, ge=1, le=10)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

_SECTION_CATEGORIES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work", "background": "related_work",
    "prior work": "related_work", "literature": "related_work",
    "method": "methodology", "methodology": "methodology",
    "approach": "methodology", "model": "methodology",
    "framework": "methodology", "system": "methodology",
    "proposed": "methodology", "architecture": "methodology",
    "experiment": "experiments", "experiments": "experiments",
    "experimental": "experiments", "setup": "experiments",
    "evaluation": "experiments", "implementation": "experiments",
    "result": "results", "results": "results",
    "analysis": "results", "discussion": "results",
    "finding": "results", "findings": "results",
    "performance": "results",
    "conclusion": "conclusion", "conclusions": "conclusion",
    "summary": "conclusion", "future work": "conclusion",
    "data": "data", "dataset": "data", "datasets": "data",
    "corpus": "data", "corpora": "data",
    "acknowledgment": "_skip", "acknowledgments": "_skip",
    "acknowledgement": "_skip", "references": "_skip",
    "appendix": "_skip", "supplementary": "_skip",
}


def _classify_section_name(header: str) -> str:
    h = re.sub(r"^\d+[\.\)]\s*", "", header).strip().lower()
    h = re.sub(r"\s+", " ", h)
    h = h.split(":::")[0].strip()
    if h in _SECTION_CATEGORIES:
        return _SECTION_CATEGORIES[h]
    for pattern, category in _SECTION_CATEGORIES.items():
        if pattern in h:
            return category
    return "content"


def parse_sections(full_text: str) -> List[tuple]:
    if full_text.count("\\n") > full_text.count("\n"):
        full_text = full_text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    parts = re.split(r"\n\n+", full_text.strip())
    sections: List[tuple] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if len(header) < 80 and body and not header.endswith("."):
            category = _classify_section_name(header)
            sections.append((category, header, body))
        else:
            if sections:
                prev_cat, prev_hdr, prev_body = sections[-1]
                sections[-1] = (prev_cat, prev_hdr, prev_body + " " + part)
            else:
                sections.append(("abstract", "", part))
    return sections


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_body(body: str) -> List[str]:
    sents = split_sentences(body)
    chunks: List[str] = []
    i = 0
    while i < len(sents):
        buf: List[str] = []
        size = 0
        j = i
        while j < len(sents):
            s = sents[j]
            if size and size + len(s) + 1 > CHILD_MAX_CHARS:
                break
            buf.append(s)
            size += len(s) + 1
            j += 1
            if size >= CHILD_TARGET_CHARS:
                break
        if not buf:
            break
        chunks.append(" ".join(buf))
        if j >= len(sents):
            break
        i = max(j - SENTENCE_OVERLAP, i + 1)
    return chunks


def build_chunks(text: str) -> List[str]:
    sections = parse_sections(text)
    chunks: List[str] = []
    for cat, _hdr, body in sections:
        if cat == "_skip":
            continue
        chunks.extend(_chunk_body(body))
    if not chunks:
        chunks = _chunk_body(text)
    return chunks


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        self.idf = {
            t: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

    def score(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.N
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            denom_norm = 1 - self.b + self.b * dl / max(self.avgdl, 1)
            s = 0.0
            for q in query:
                if q not in tf:
                    continue
                idf = self.idf.get(q, 0.0)
                f = tf[q]
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * denom_norm)
            scores[i] = s
        return scores


def retrieve(chunks: List[str], question: str, top_k: int) -> List[Dict[str, Any]]:
    tokenized = [tokenize(c) for c in chunks]
    bm25 = BM25(tokenized)
    scores = bm25.score(tokenize(question))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    pool = ranked[:BM25_TOP_K]
    final = pool[:top_k]
    return [
        {"chunk_id": idx, "score": round(float(sc), 4), "text": chunks[idx]}
        for idx, sc in final
    ]


def build_prompt(question: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    ctx = "\n\n".join(
        f"[Chunk {e['chunk_id']}] {e['text']}" for e in evidence
    )
    system = (
        "You are a precise research QA assistant. Answer the question using ONLY "
        "the provided evidence chunks. If the evidence is insufficient, say so. "
        "Keep the answer concise (1-3 sentences) and factual."
    )
    user = f"Evidence:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(messages: List[Dict[str, str]]) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY not configured on the server.",
        )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 400,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(OPENROUTER_URL, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e.response.text[:300]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "GenAI Research QA Agent",
        "model": OPENROUTER_MODEL,
        "llm_configured": bool(OPENROUTER_API_KEY),
    }


@app.get("/")
def index():
    html = STATIC_DIR / "index.html"
    if html.exists():
        return FileResponse(str(html))
    return {"status": "ok", "service": "GenAI Research QA Agent"}


@app.post("/query")
def query(request: QueryRequest):
    t0 = time.time()
    chunks = build_chunks(request.paper_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract any text chunks from paper_text.")
    t_chunk = time.time() - t0

    t1 = time.time()
    retrieved = retrieve(chunks, request.question, request.top_k)
    t_retrieve = time.time() - t1

    t2 = time.time()
    messages = build_prompt(request.question, retrieved)
    answer = call_llm(messages)
    t_llm = time.time() - t2

    return {
        "answer": answer,
        "evidence": [{"chunk_id": e["chunk_id"], "score": e["score"], "text": e["text"]} for e in retrieved],
        "trace": {
            "total_chunks": len(chunks),
            "retrieved_chunk_ids": [e["chunk_id"] for e in retrieved],
            "retrieval_scores": [e["score"] for e in retrieved],
            "model": OPENROUTER_MODEL,
            "timings_sec": {
                "chunking": round(t_chunk, 3),
                "retrieval": round(t_retrieve, 3),
                "llm": round(t_llm, 3),
                "total": round(time.time() - t0, 3),
            },
        },
    }
