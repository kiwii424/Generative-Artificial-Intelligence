#!/usr/bin/env python3
"""
HW2: Document QA based on RAG
Student ID: 111511157

Pipeline:
  Sentence-based Parent-Child Chunking
  → Hybrid Retrieval (Dense FAISS + BM25)
  → RRF Fusion → Cross-Encoder Reranking (on parent_text for accuracy)
  → LLM Answer Generation (Llama-3.2-3B via OpenRouter)

Usage:
  uv run python 111511157.py                                                # Process private_dataset.json
  uv run python 111511157.py --eval --output public.json                    # Evaluate on public_dataset.json (random sample)
  uv run python 111511157.py --eval --sample 0   --output public.json       # Evaluate all papers (no sampling)
  uv run python 111511157.py --dataset path.json                            # Custom dataset
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import faiss
import nltk
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from dotenv import load_dotenv

# 載入 .env 變數
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("no api key found. Please set OPENROUTER_API_KEY in your environment variables.")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ═══════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════════════════════
def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Two handlers:
      - FileHandler  → DEBUG+  (all stage details)
      - StreamHandler → WARNING+ (terminal: only errors / critical info)
    tqdm.write() is used for terminal progress messages to avoid line-break conflicts.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    logger = logging.getLogger("rag")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # File handler — everything
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)

    # Console handler — WARNING and above only (errors / API failures)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    return logger, log_path


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "meta-llama/llama-3.2-3b-instruct"
LLM_BASE_URL = "https://openrouter.ai/api/v1"


# 切塊越小，初步檢索的 Top K 通常需要調大
# Parent 視窗越大，餵給 LLM 的 Chunk 數量就應該越少
# RERANK_POOL 必須大於你期望 Dynamic K 選出的最大數量，才能確保精選品質


# Chunking — evidence median=216 chars, p75=383
CHILD_TARGET_CHARS = 200
CHILD_MAX_CHARS = 400
PARENT_WINDOW_CHUNKS = 2
PARENT_MAX_CHARS = 800

# Retrieval
DENSE_TOP_K = 40
BM25_TOP_K = 40
RRF_TOP_K = 20
RRF_K = 60
DEFAULT_FINAL_K = 3
RERANK_POOL = 15

# Dynamic K selection
RERANKER_SCORE_THRESHOLD = -1.0
RERANKER_GAP_THRESHOLD = 5.0

# LLM
MAX_EVIDENCE_FOR_LLM = 5
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 256

# Eval sampling: number of random papers to evaluate (0 = all)
EVAL_SAMPLE_N = 20


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Chunk:
    chunk_id: int
    text: str          # child chunk — submitted as evidence (small, high ROUGE-L)
    parent_text: str   # parent window — used for reranking + LLM context (richer)
    sent_start: int
    sent_end: int


@dataclass
class QAResult:
    title: str
    answer: str
    evidence: list  # list[str] of child chunk texts


# ═══════════════════════════════════════════════════════════════════════════════
# DocumentProcessor — sentence-aligned parent-child chunking
# ═══════════════════════════════════════════════════════════════════════════════
class DocumentProcessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.child_target = CHILD_TARGET_CHARS
        self.child_max = CHILD_MAX_CHARS
        self.parent_window = PARENT_WINDOW_CHUNKS
        self.parent_max = PARENT_MAX_CHARS

    def split_sentences(self, text: str) -> list:
        """Split text into sentences, merging very short fragments."""
        raw = nltk.sent_tokenize(text)
        merged = []
        buf = ""
        for s in raw:
            s = s.strip()
            if not s:
                continue
            if buf:
                buf = buf + " " + s
                if len(buf) >= 60:
                    merged.append(buf)
                    buf = ""
            elif len(s) < 60:
                buf = s
            else:
                merged.append(s)
        if buf:
            if merged:
                merged[-1] = merged[-1] + " " + buf
            else:
                merged.append(buf)
        return merged

    def build_chunks(self, full_text: str) -> list:
        """Build parent-child chunks from full_text."""
        sentences = self.split_sentences(full_text)
        if not sentences:
            return [Chunk(0, full_text[: self.child_max], full_text[: self.parent_max], 0, 0)]

        # Group sentences into child chunks (~200 chars each)
        child_groups = []  # (sent_start, sent_end, text)
        current_text = ""
        start_idx = 0

        for i, sent in enumerate(sentences):
            candidate = (current_text + " " + sent).strip() if current_text else sent
            if current_text and len(candidate) > self.child_target:
                child_groups.append((start_idx, i - 1, current_text))
                current_text = sent
                start_idx = i
            else:
                current_text = candidate

        if current_text:
            child_groups.append((start_idx, len(sentences) - 1, current_text))

        # Force-split oversized chunks at sentence/clause boundaries
        final_groups = []
        for s, e, text in child_groups:
            if len(text) <= self.child_max:
                final_groups.append((s, e, text))
            else:
                mid = len(text) // 2
                sp = text.rfind(". ", mid - 80, mid + 80)
                if sp == -1:
                    sp = text.rfind(", ", mid - 80, mid + 80)
                if sp == -1:
                    sp = mid
                final_groups.append((s, e, text[: sp + 1].strip()))
                final_groups.append((s, e, text[sp + 1 :].strip()))

        # Attach parent windows (2 chunks before + current + 2 after)
        chunks = []
        for idx, (s, e, text) in enumerate(final_groups):
            pw_lo = max(0, idx - self.parent_window)
            pw_hi = min(len(final_groups), idx + self.parent_window + 1)
            parent = " ".join(g[2] for g in final_groups[pw_lo:pw_hi])
            if len(parent) > self.parent_max:
                parent = parent[: self.parent_max]
            chunks.append(Chunk(chunk_id=idx, text=text, parent_text=parent,
                                sent_start=s, sent_end=e))

        # Log each chunk to file
        self.logger.debug(f"[Chunking] {len(chunks)} chunks (target={self.child_target}, max={self.child_max}):")
        for c in chunks:
            self.logger.debug(f"  #{c.chunk_id:>3}  {len(c.text):>4}c  {c.text[:80]!r}")

        return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Retriever — hybrid dense+BM25, RRF fusion, cross-encoder reranking
# ═══════════════════════════════════════════════════════════════════════════════
class Retriever:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        tqdm.write(f"[Init] Loading embedding model: {EMBED_MODEL}")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        tqdm.write(f"[Init] Loading reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=1024)
        tqdm.write("[Init] Models loaded.\n")
        self.chunks = []
        self.index = None
        self.bm25 = None

    def build_index(self, chunks: list):
        """Build FAISS + BM25 indexes for one paper's chunks."""
        self.chunks = chunks
        if not chunks:
            return
        texts = [c.text for c in chunks]

        # Dense: BGE embeddings → FAISS inner-product (cosine on L2-normed vectors)
        embs = self.embed_model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs.astype("float32"))

        # Sparse: BM25
        self.bm25 = BM25Okapi([self._tok(t) for t in texts])

        self.logger.debug(f"[Indexing] {self.index.ntotal} FAISS vectors + BM25 ({len(texts)} docs)")

    @staticmethod
    def _tok(text: str) -> list:
        return re.findall(r"\w+", text.lower())

    def _embed_query(self, q: str):
        prefixed = "Represent this sentence for searching relevant passages: " + q
        return self.embed_model.encode([prefixed], normalize_embeddings=True).astype("float32")

    def dense_search(self, query: str, top_k: int) -> list:
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(self._embed_query(query), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def _embed_passage(self, text: str):
        """Embed text as a passage (no query prefix) — used for HyDE."""
        return self.embed_model.encode([text], normalize_embeddings=True).astype("float32")

    def hyde_search(self, hyde_text: str, top_k: int) -> list:
        """Search using HyDE embedding (passage-style, no query prefix)."""
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        emb = self._embed_passage(hyde_text)
        scores, ids = self.index.search(emb, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def bm25_search(self, query: str, top_k: int) -> list:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(self._tok(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(i, float(scores[i])) for i in ranked]

    @staticmethod
    def rrf_fuse(*rank_lists, k=RRF_K) -> list:
        scores = defaultdict(float)
        for rl in rank_lists:
            for rank, (doc_id, _) in enumerate(rl):
                scores[doc_id] += 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda x: -scores[x])

    def rerank(self, query: str, ids: list, top_k: int) -> list:
        """Rerank using PARENT_TEXT for richer semantic context."""
        if not ids:
            return []
        pairs = [(query, self.chunks[i].parent_text) for i in ids]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(ids, scores), key=lambda x: -x[1])
        return ranked[:top_k]

    def select_dynamic_k(self, reranked: list, max_k: int) -> list:
        """Dynamically select evidence count based on reranker score distribution."""
        if not reranked or len(reranked) <= 1:
            return reranked[:1] if reranked else []

        selected = [reranked[0]]  # always keep the best chunk

        for i in range(1, min(len(reranked), max_k)):
            chunk_id, score = reranked[i]
            prev_score = reranked[i - 1][1]

            # Stop if score drops below absolute threshold
            if score < RERANKER_SCORE_THRESHOLD:
                break

            # Stop if gap from previous chunk is too large
            if prev_score - score > RERANKER_GAP_THRESHOLD:
                break

            selected.append(reranked[i])

        self.logger.debug(f"[DynamicK] max_k={max_k}, selected={len(selected)}")
        return selected

    def retrieve(self, question: str, variants: list, final_k: int, hyde_text: str = "") -> tuple:
        """Full pipeline: multi-query dense+BM25 (+HyDE) → RRF → rerank → dynamic-K."""
        if not self.chunks:
            return [], {}

        # Stage 1: Dense + BM25 retrieval across all query variants
        all_dense, all_bm25 = [], []
        for q in variants:
            d = self.dense_search(q, DENSE_TOP_K)
            b = self.bm25_search(q, BM25_TOP_K)
            all_dense.extend(d)
            all_bm25.extend(b)
            self.logger.debug(f"  [Query] {q!r:70s}  dense={len(d)}, bm25={len(b)}")

        # HyDE: embed hypothetical answer as a passage (no query prefix)
        if hyde_text:
            h = self.hyde_search(hyde_text, DENSE_TOP_K)
            all_dense.extend(h)
            self.logger.debug(f"  [HyDE] dense results: {len(h)}")

        dense_unique = len(set(i for i, _ in all_dense))
        bm25_unique = len(set(i for i, _ in all_bm25))

        # Stage 2: RRF fusion
        fused = self.rrf_fuse(all_dense, all_bm25)[:RRF_TOP_K]
        if not fused:
            return self.chunks[:final_k], {"fallback": True}

        self.logger.debug(f"[RRF] dense_unique={dense_unique}, bm25_unique={bm25_unique}, fused={len(fused)}")

        # Stage 3: Cross-encoder reranking with larger pool
        pool_k = min(RERANK_POOL, len(fused))
        reranked = self.rerank(question, fused, pool_k)

        # Stage 4: Dynamic K selection based on reranker scores
        selected = self.select_dynamic_k(reranked, final_k)

        self.logger.debug(f"[Rerank] pool={pool_k}, selected={len(selected)}:")
        for chunk_id, score in selected:
            self.logger.debug(f"  #{chunk_id:>3}  score={score:.4f}  {self.chunks[chunk_id].text[:80]!r}")

        debug = {
            "dense_unique": dense_unique,
            "bm25_unique": bm25_unique,
            "rrf_candidates": len(fused),
            "rerank_pool": pool_k,
            "rerank_scores": [(i, round(s, 4)) for i, s in reranked],
            "final_k": len(selected),
        }
        return [self.chunks[i] for i, _ in selected], debug

    def clear(self):
        self.chunks, self.index, self.bm25 = [], None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Generator — Llama-3.2-3B via OpenRouter
# ═══════════════════════════════════════════════════════════════════════════════
class Generator:
    def __init__(self, api_key: str, logger: logging.Logger):
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
        self.logger = logger

    def generate_hyde(self, title: str, question: str) -> str:
        """Generate hypothetical document passage for HyDE retrieval."""
        messages = [
            {"role": "system", "content":
             "You are a scientific paper expert. Given a paper title and question, "
             "write a brief passage (2-3 sentences) that might appear in the paper "
             "as an answer. Write as if quoting the paper. Be specific and technical."},
            {"role": "user", "content": f"Paper: {title}\nQuestion: {question}\nPassage:"}
        ]
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL, messages=messages,
                temperature=0.3, max_tokens=150,
            )
            result = resp.choices[0].message.content.strip()
            self.logger.debug(f"[HyDE] Generated: {result[:200]}")
            return result
        except Exception as e:
            self.logger.warning(f"[HyDE] Failed: {e}")
            return ""

    def generate(self, title: str, question: str, chunks: list, max_retries: int = 5) -> tuple:
        """Returns (answer_str, debug_info)."""
        system = (
            "You are a precise question-answering assistant for scientific papers. "
            "Answer ONLY from the provided evidence passages. Do not use outside knowledge. "
            "Be concise and precise. Include specific names, numbers, and technical terms from the evidence. "
            "Answer in 1-3 sentences unless the question requires listing multiple items."
        )

        evidence = chunks[:MAX_EVIDENCE_FOR_LLM]
        ev_lines = []
        total_ev_chars = 0
        for i, c in enumerate(evidence, 1):
            pt = c.parent_text[:PARENT_MAX_CHARS]
            ev_lines.append(f"[{i}] {pt}")
            total_ev_chars += len(pt)

        user = (
            f"Paper: {title}\n\n"
            f"Evidence passages:\n" + "\n".join(ev_lines) + "\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the evidence above:"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        prompt_chars = len(system) + len(user)
        est_tokens = prompt_chars // 4

        self.logger.debug(f"[LLM] n_evidence={len(evidence)}, ev_chars={total_ev_chars}, "
                          f"prompt_chars={prompt_chars}, est_tokens={est_tokens}")
        self.logger.debug(f"[LLM] prompt_user=\n{user}")

        debug = {
            "n_evidence_for_llm": len(evidence),
            "evidence_total_chars": total_ev_chars,
            "prompt_chars": prompt_chars,
            "est_prompt_tokens": est_tokens,
        }

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                answer = resp.choices[0].message.content.strip()
                if resp.usage:
                    debug["actual_prompt_tokens"] = resp.usage.prompt_tokens
                    debug["completion_tokens"] = resp.usage.completion_tokens
                    self.logger.debug(f"[LLM] actual: {resp.usage.prompt_tokens} prompt + "
                                      f"{resp.usage.completion_tokens} completion tokens")
                self.logger.debug(f"[LLM] answer: {answer}")
                return answer, debug
            except Exception as e:
                wait = min(2 ** (attempt + 1), 30)
                self.logger.warning(f"[LLM] Error (attempt {attempt+1}/{max_retries}): {e} — retry in {wait}s")
                time.sleep(wait)

        self.logger.error("[LLM] All retries failed, returning fallback answer.")
        return "Unable to generate answer.", debug


# ═══════════════════════════════════════════════════════════════════════════════
# Query Variant Generator (rule-based, no LLM cost)
# ═══════════════════════════════════════════════════════════════════════════════
_STOP_WORDS = frozenset(
    "the and for are was were has have had does did can could would should "
    "this that these those with from into what which how why where when who "
    "whom whose not been about also more than their they them some other "
    "will may must shall being used using".split()
)


def generate_query_variants(question: str) -> list:
    variants = [question]

    stripped = re.sub(
        r"^(what|which|how|why|where|when|who|does|do|did|is|are|was|were|can|could)"
        r"\s+(is|are|was|were|does|do|did|the|a|an|this|that)?\s*",
        "",
        question,
        flags=re.IGNORECASE,
    ).strip().rstrip("?").strip()
    if stripped and stripped.lower() != question.lower() and len(stripped) > 10:
        variants.append(stripped)

    words = re.findall(r"\b[a-zA-Z]{3,}\b", question)
    kw = [w for w in words if w.lower() not in _STOP_WORDS]
    if len(kw) >= 2:
        variants.append(" ".join(kw))

    return variants[:3]


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
class RAGPipeline:
    def __init__(self, api_key: str, logger: logging.Logger):
        self.logger = logger
        self.processor = DocumentProcessor(logger)
        self.retriever = Retriever(logger)
        self.generator = Generator(api_key=api_key, logger=logger)

    @staticmethod
    def determine_k(question: str) -> int:
        q = question.lower()
        list_kw = [
            "list", "what are the", "which", "name the", "how many",
            "what methods", "what techniques", "what datasets",
            "what approaches", "what models", "what features",
        ]
        return 5 if any(k in q for k in list_kw) else DEFAULT_FINAL_K

    def process_paper(self, entry: dict) -> QAResult:
        title = entry["title"]
        full_text = entry["full_text"]
        question = entry["question"]

        self.logger.debug(f"\n{'='*70}")
        self.logger.debug(f"Paper: {title}")
        self.logger.debug(f"Question: {question}")
        self.logger.debug(f"Text length: {len(full_text):,} chars")
        t_start = time.time()

        # Stage 1: Chunking
        t0 = time.time()
        chunks = self.processor.build_chunks(full_text)
        dt_chunk = time.time() - t0
        chunk_lens = [len(c.text) for c in chunks]
        avg_len = sum(chunk_lens) / len(chunk_lens) if chunk_lens else 0
        self.logger.debug(f"[Chunking] {len(chunks)} chunks in {dt_chunk:.2f}s, "
                          f"avg={avg_len:.0f}, min={min(chunk_lens)}, max={max(chunk_lens)}")

        # Stage 2: Indexing
        t0 = time.time()
        self.retriever.build_index(chunks)
        dt_index = time.time() - t0
        self.logger.debug(f"[Indexing] done in {dt_index:.2f}s")

        # Stage 3: Query Variants
        variants = generate_query_variants(question)
        self.logger.debug(f"[Query variants] ({len(variants)}):")
        for i, v in enumerate(variants):
            self.logger.debug(f"  Q{i}: {v}")

        # Stage 3b: HyDE (Hypothetical Document Embeddings)
        t0 = time.time()
        hyde_text = self.generator.generate_hyde(title, question)
        dt_hyde = time.time() - t0
        self.logger.debug(f"[HyDE] generated in {dt_hyde:.2f}s")

        # Stage 4: Retrieval
        k = self.determine_k(question)
        t0 = time.time()
        retrieved, retrieval_debug = self.retriever.retrieve(question, variants, k, hyde_text=hyde_text)
        dt_retrieval = time.time() - t0
        self.logger.debug(f"[Retrieval] K={k}, got {len(retrieved)} chunks in {dt_retrieval:.2f}s")

        # Stage 5: LLM Generation
        t0 = time.time()
        answer, gen_debug = self.generator.generate(title, question, retrieved)
        dt_llm = time.time() - t0
        self.logger.debug(f"[LLM] generated in {dt_llm:.2f}s")

        # Stage 6: Cleanup
        self.retriever.clear()

        total_time = time.time() - t_start
        evidence = [c.text for c in retrieved]

        self.logger.debug(f"[Result] answer ({len(answer)} chars): {answer}")
        self.logger.debug(f"[Result] evidence: {len(evidence)} chunks")
        self.logger.debug(f"[Time] total={total_time:.1f}s "
                          f"(chunk={dt_chunk:.1f}s, index={dt_index:.1f}s, "
                          f"hyde={dt_hyde:.1f}s, retrieval={dt_retrieval:.1f}s, llm={dt_llm:.1f}s)")

        return QAResult(title=title, answer=answer, evidence=evidence)

    def run(self, dataset: list) -> list:
        results = []
        t_all = time.time()

        pbar = tqdm(dataset, desc="Processing papers", unit="paper",
                    dynamic_ncols=True, file=sys.stdout)
        for entry in pbar:
            short_title = entry["title"][:45] + "..." if len(entry["title"]) > 45 else entry["title"]
            pbar.set_postfix_str(short_title)
            result = self.process_paper(entry)
            results.append(result)
            # Print answer preview to terminal (immediately useful)
            ans_preview = result.answer[:100].replace("\n", " ")
            tqdm.write(f"  → {ans_preview}{'...' if len(result.answer) > 100 else ''}")

        elapsed = time.time() - t_all
        n = len(dataset)
        tqdm.write(f"\nDone: {n} papers in {elapsed:.1f}s ({elapsed/n:.1f}s/paper avg)")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator — local ROUGE-L check (matches score_public.py formula exactly)
# ═══════════════════════════════════════════════════════════════════════════════
class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def evidence_score_single(self, retrieved: list, golden: list) -> float:
        if not golden:
            return 1.0
        if not retrieved:
            return 0.0
        total = 0.0
        for chunk in retrieved:
            result = self.scorer.score_multi(targets=golden, prediction=chunk)
            total += result["rougeL"].fmeasure
        return total / len(retrieved)

    def evaluate(self, results: list, dataset: list) -> dict:
        gt_map = {item["title"]: item for item in dataset}
        ev_scores = []

        tqdm.write(f"\n{'='*70}")
        tqdm.write("  Evidence Score Evaluation (ROUGE-L, same formula as TA)")
        tqdm.write(f"{'='*70}")

        for r in results:
            gt = gt_map.get(r.title)
            if gt is None:
                tqdm.write(f"  [SKIP] {r.title[:50]} — not in ground truth")
                continue
            golden_ev = gt.get("evidence", [])
            s = self.evidence_score_single(r.evidence, golden_ev)
            ev_scores.append(s)
            tqdm.write(f"  {s:.4f}  K={len(r.evidence):>2}  gt_ev={len(golden_ev)}  {r.title[:50]}")

        if not ev_scores:
            tqdm.write("  No papers scored.")
            return {"mean_evidence_score": 0, "n": 0}

        avg = sum(ev_scores) / len(ev_scores)
        tqdm.write(f"\n  {'─'*60}")
        tqdm.write(f"  Mean Evidence Score : {avg:.5f}  (n={len(ev_scores)})")
        tqdm.write(f"  Weak baseline       : 0.2124")
        tqdm.write(f"  Strong baseline     : 0.26185")
        if avg > 0.26185:
            tqdm.write(f"  ✓ Above Strong Baseline!")
        elif avg > 0.2124:
            tqdm.write(f"  ✓ Above Weak Baseline")
        else:
            tqdm.write(f"  ✗ Below Weak Baseline — needs improvement")
        tqdm.write(f"  {'─'*60}")

        return {"mean_evidence_score": avg, "n": len(ev_scores), "scores": ev_scores}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="HW2 RAG Pipeline — 111511157")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset JSON")
    parser.add_argument("--eval", action="store_true", help="Evaluate on public_dataset.json")
    parser.add_argument("--sample", type=int, default=EVAL_SAMPLE_N,
                        help=f"Papers to randomly sample in eval mode (0=all, default={EVAL_SAMPLE_N})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    STUDENT_ID = "111511157"
    script_dir = Path(__file__).resolve().parent

    # Setup logging
    log_dir = script_dir / "logs"
    logger, log_path = setup_logging(log_dir)

    # Resolve dataset path
    if args.dataset:
        dataset_path = Path(args.dataset)
    elif args.eval:
        dataset_path = script_dir / "public_dataset.json"
        if not dataset_path.exists():
            dataset_path = script_dir / "datasets" / "public_dataset.json"
    else:
        dataset_path = script_dir / "private_dataset.json"
        if not dataset_path.exists():
            dataset_path = script_dir / "datasets" / "private_dataset.json"

    output_path = Path(args.output) if args.output else script_dir / f"{STUDENT_ID}.json"

    # API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "ollama")
    if not api_key:
        tqdm.write("ERROR: Set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    # Config banner
    tqdm.write(f"{'#'*70}")
    tqdm.write(f"# HW2 RAG Pipeline — Student {STUDENT_ID}")
    tqdm.write(f"{'#'*70}")
    tqdm.write(f"  Dataset  : {dataset_path}")
    tqdm.write(f"  Output   : {output_path}")
    tqdm.write(f"  Log file : {log_path}")
    tqdm.write(f"  Eval mode: {args.eval}  (sample={args.sample if args.sample > 0 else 'all'})")
    tqdm.write(f"  Embed    : {EMBED_MODEL}")
    tqdm.write(f"  Reranker : {RERANKER_MODEL}")
    tqdm.write(f"  LLM      : {LLM_MODEL}")
    tqdm.write(f"  Chunk    : target={CHILD_TARGET_CHARS}, max={CHILD_MAX_CHARS} chars")
    tqdm.write(f"  Parent   : window={PARENT_WINDOW_CHUNKS}, max={PARENT_MAX_CHARS} chars")
    tqdm.write(f"  Retrieval: dense_k={DENSE_TOP_K}, bm25_k={BM25_TOP_K}, "
               f"rrf_k={RRF_TOP_K}, final_k={DEFAULT_FINAL_K}")
    tqdm.write(f"  LLM ctx  : max {MAX_EVIDENCE_FOR_LLM} parent chunks, {PARENT_MAX_CHARS} chars each")
    tqdm.write("")

    logger.debug(f"Config: embed={EMBED_MODEL}, reranker={RERANKER_MODEL}, llm={LLM_MODEL}")
    logger.debug(f"Chunk: target={CHILD_TARGET_CHARS}, max={CHILD_MAX_CHARS}, "
                 f"parent_window={PARENT_WINDOW_CHUNKS}, parent_max={PARENT_MAX_CHARS}")

    # Load dataset
    if not dataset_path.exists():
        tqdm.write(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path) as f:
        dataset = json.load(f)
    tqdm.write(f"Loaded {len(dataset)} papers from {dataset_path.name}")

    # Random sampling for eval
    if args.eval and args.sample > 0 and args.sample < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.sample)
        tqdm.write(f"Sampled {len(dataset)} papers (seed={args.seed})")
    tqdm.write("")

    # Run pipeline
    pipeline = RAGPipeline(api_key=api_key, logger=logger)
    results = pipeline.run(dataset)

    # Save results (only when running on full private dataset, not eval sample)
    if not args.eval:
        output = [{"title": r.title, "answer": r.answer, "evidence": r.evidence} for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        tqdm.write(f"\nSaved {len(output)} results → {output_path}")

        # Validate format
        errors = []
        for r in output:
            if not isinstance(r["answer"], str) or not r["answer"].strip():
                errors.append(f"  Empty/invalid answer for: {r['title'][:50]}")
            n_ev = len(r["evidence"])
            if n_ev < 1 or n_ev > 40:
                errors.append(f"  Evidence count {n_ev} (need 1-40) for: {r['title'][:50]}")
        if errors:
            tqdm.write("\nFORMAT ERRORS:")
            for e in errors:
                tqdm.write(e)
            sys.exit(1)
        else:
            tqdm.write("Format validation: PASSED")

    # Evaluate if --eval
    if args.eval:
        evaluator = Evaluator()
        evaluator.evaluate(results, dataset)

    tqdm.write(f"\nLog saved → {log_path}")


if __name__ == "__main__":
    main()