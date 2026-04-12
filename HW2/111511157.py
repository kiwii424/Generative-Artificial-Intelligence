#!/usr/bin/env python3
"""
HW2: Document QA based on RAG


Default LLM provider: openrouter
For local Ollama, prefix commands with LLM_PROVIDER=ollama

Usage:
  uv run python 111511157.py                                                    # Process private_dataset.json
  uv run python 111511157.py --eval --output ./public.json                      # Evaluate on public_dataset.json (random sample)
  uv run python 111511157.py --eval --sample 0 --output ./public.json           # Evaluate all papers (no sampling)
  uv run python 111511157.py --dataset path.json                                # Custom dataset
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
from dataclasses import dataclass, field
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
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    logger = logging.getLogger("rag")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    # fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)

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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:3b" if LLM_PROVIDER == "ollama" else "meta-llama/llama-3.2-3b-instruct",
)
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "http://127.0.0.1:11434/v1" if LLM_PROVIDER == "ollama" else "https://openrouter.ai/api/v1",
)
LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    if LLM_PROVIDER == "ollama":
        LLM_API_KEY = "ollama"
    else:
        LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")

if LLM_PROVIDER == "openrouter" and not LLM_API_KEY:
    print("no api key found. Please set OPENROUTER_API_KEY or LLM_API_KEY in your environment variables.")


# Chunking — evidence median=216 chars, p75=383
CHILD_TARGET_CHARS = 200
CHILD_MAX_CHARS = 400
PARENT_WINDOW_CHUNKS = 2
PARENT_MAX_CHARS = 800
CHUNK_SENTENCE_OVERLAP = 1  # NEW: 1-sentence overlap between consecutive chunks

# Retrieval
DENSE_TOP_K = 40
BM25_TOP_K = 40
RRF_TOP_K = 20
RRF_K = 60
DEFAULT_FINAL_K = 2          # CHANGED: was 3, reduced to avoid filler evidence
RERANK_POOL = 15

# Dynamic K selection — TIGHTENED significantly
RERANKER_SCORE_THRESHOLD = 0.0    # CHANGED: was -1.0 — only keep positively-scored chunks
RERANKER_GAP_THRESHOLD = 2.0      # CHANGED: was 5.0 — drop if score gap is large

# Section boosting
SECTION_BOOST_SCORE = 1.5  # NEW: bonus added to reranker score for matching sections

# Embedding query prefix (model-specific — must match embedding model's training format)
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge-*
EMBED_PASSAGE_PREFIX = ""  # bge-* doesn't need passage prefix

# LLM
MAX_EVIDENCE_FOR_LLM = 5
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 256

# Eval sampling
EVAL_SAMPLE_N = 20


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Chunk:
    chunk_id: int
    text: str          # child chunk — submitted as evidence
    parent_text: str   # parent window — used for reranking + LLM context
    sent_start: int
    sent_end: int
    section: str = ""  # NEW: section name (e.g. "abstract", "methodology", "results")


@dataclass
class QAResult:
    title: str
    answer: str
    evidence: list


# ═══════════════════════════════════════════════════════════════════════════════
# Section Parser — extract paper structure
# ═══════════════════════════════════════════════════════════════════════════════
# Known section name patterns (normalized to lowercase categories)
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
    """Map a section header to a normalized category."""
    h = re.sub(r"^\d+[\.\)]\s*", "", header).strip().lower()
    h = re.sub(r"\s+", " ", h)
    # Exact match first
    if h in _SECTION_CATEGORIES:
        return _SECTION_CATEGORIES[h]
    # Substring match
    for pattern, category in _SECTION_CATEGORIES.items():
        if pattern in h:
            return category
    return "content"  # Unknown sections default to "content" (not skipped)


def parse_sections(full_text: str) -> list:
    """
    Parse paper full_text into (section_name, section_text) pairs.
    Papers are structured as: Header\\nContent\\n\\nHeader\\nContent\\n\\n...
    """
    parts = re.split(r"\n\n+", full_text.strip())
    sections = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        # Classify: if header is short and looks like a section title
        if len(header) < 80 and body and not header.endswith("."):
            category = _classify_section_name(header)
            sections.append((category, header, body))
        else:
            # No clear header — treat entire part as continuation of previous section
            combined = part
            if sections:
                prev_cat, prev_hdr, prev_body = sections[-1]
                sections[-1] = (prev_cat, prev_hdr, prev_body + " " + combined)
            else:
                sections.append(("abstract", "", combined))

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Query Router — classify question to target sections
# ═══════════════════════════════════════════════════════════════════════════════
_QUESTION_SECTION_MAP = [
    # (keywords_in_question, target_section_categories)
    (["method", "approach", "technique", "algorithm", "how do they",
      "how does the", "how is", "how are", "procedure", "pipeline",
      "architecture", "framework", "implement", "design", "propose"],
     {"methodology"}),
    (["result", "performance", "accuracy", "score", "f1", "bleu", "rouge",
      "improve", "outperform", "baseline", "benchmark", "achieve", "gain",
      "state-of-the-art", "sota", "compare", "comparison"],
     {"results", "experiments"}),
    (["dataset", "data", "corpus", "corpora", "benchmark", "annotation",
      "annotate", "label", "training data", "test set"],
     {"data", "experiments"}),
    (["conclusion", "finding", "discover", "observe", "limitation",
      "future work", "contribution"],
     {"conclusion", "results"}),
    (["background", "related", "prior work", "previous", "existing"],
     {"related_work", "introduction"}),
    (["abstract", "summary", "overview", "main idea", "key contribution"],
     {"abstract", "introduction"}),
]


def classify_question_sections(question: str) -> set:
    """Classify a question to a set of target section categories."""
    q = question.lower()
    targets = set()
    for keywords, sections in _QUESTION_SECTION_MAP:
        if any(kw in q for kw in keywords):
            targets.update(sections)
    return targets


# ═══════════════════════════════════════════════════════════════════════════════
# DocumentProcessor — section-aware chunking with sentence overlap
# ═══════════════════════════════════════════════════════════════════════════════
class DocumentProcessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.child_target = CHILD_TARGET_CHARS
        self.child_max = CHILD_MAX_CHARS
        self.parent_window = PARENT_WINDOW_CHUNKS
        self.parent_max = PARENT_MAX_CHARS
        self.sentence_overlap = CHUNK_SENTENCE_OVERLAP

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

    def _group_sentences_to_chunks(self, sentences: list) -> list:
        """Group sentences into child chunks with optional overlap."""
        if not sentences:
            return []

        child_groups = []  # (sent_start, sent_end, text)
        current_sents = []
        start_idx = 0

        for i, sent in enumerate(sentences):
            current_text = " ".join(current_sents + [sent])
            if current_sents and len(current_text) > self.child_target:
                # Finalize current chunk
                child_groups.append((start_idx, i - 1, " ".join(current_sents)))
                # Start new chunk with overlap
                overlap_start = max(0, len(current_sents) - self.sentence_overlap)
                overlap_sents = current_sents[overlap_start:]
                current_sents = overlap_sents + [sent]
                start_idx = i - len(overlap_sents)
            else:
                current_sents.append(sent)

        if current_sents:
            child_groups.append((start_idx, len(sentences) - 1, " ".join(current_sents)))

        # Force-split oversized chunks
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

        return final_groups

    def build_chunks(self, full_text: str) -> list:
        """Build section-aware parent-child chunks from full_text."""
        sections = parse_sections(full_text)
        all_groups = []  # (sent_start, sent_end, text, section_category)

        for section_cat, section_header, section_body in sections:
            if section_cat == "_skip":
                continue  # Skip acknowledgments, references, etc.

            sentences = self.split_sentences(section_body)
            if not sentences:
                continue

            groups = self._group_sentences_to_chunks(sentences)
            for s, e, text in groups:
                all_groups.append((s, e, text, section_cat))

        if not all_groups:
            # Fallback: process entire text without section awareness
            sentences = self.split_sentences(full_text)
            if not sentences:
                return [Chunk(0, full_text[:self.child_max], full_text[:self.parent_max], 0, 0)]
            groups = self._group_sentences_to_chunks(sentences)
            for s, e, text in groups:
                all_groups.append((s, e, text, "unknown"))

        # Attach parent windows
        chunks = []
        for idx, (s, e, text, section) in enumerate(all_groups):
            pw_lo = max(0, idx - self.parent_window)
            pw_hi = min(len(all_groups), idx + self.parent_window + 1)
            parent = " ".join(g[2] for g in all_groups[pw_lo:pw_hi])
            if len(parent) > self.parent_max:
                parent = parent[: self.parent_max]
            chunks.append(Chunk(
                chunk_id=idx, text=text, parent_text=parent,
                sent_start=s, sent_end=e, section=section,
            ))

        self.logger.debug(f"[Chunking] {len(chunks)} chunks from {len(sections)} sections:")
        section_counts = defaultdict(int)
        for c in chunks:
            section_counts[c.section] += 1
        for sec, cnt in sorted(section_counts.items()):
            self.logger.debug(f"  {sec}: {cnt} chunks")

        return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Retriever — hybrid dense+BM25, RRF fusion, section-boosted cross-encoder reranking
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
        self.chunks = chunks
        if not chunks:
            return
        texts = [c.text for c in chunks]

        embs = self.embed_model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs.astype("float32"))

        self.bm25 = BM25Okapi([self._tok(t) for t in texts])
        # self.logger.debug(f"[Indexing] {self.index.ntotal} FAISS vectors + BM25 ({len(texts)} docs)")

    @staticmethod
    def _tok(text: str) -> list:
        return re.findall(r"\w+", text.lower())

    def _embed_query(self, q: str):
        prefixed = EMBED_QUERY_PREFIX + q if EMBED_QUERY_PREFIX else q
        return self.embed_model.encode([prefixed], normalize_embeddings=True).astype("float32")

    def dense_search(self, query: str, top_k: int) -> list:
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(self._embed_query(query), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def _embed_passage(self, text: str):
        prefixed = EMBED_PASSAGE_PREFIX + text if EMBED_PASSAGE_PREFIX else text
        return self.embed_model.encode([prefixed], normalize_embeddings=True).astype("float32")

    def hyde_search(self, hyde_text: str, top_k: int) -> list:
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

    def rerank(self, query: str, ids: list, top_k: int, target_sections: set = None) -> list:
        """Rerank using parent_text, with optional section boosting."""
        if not ids:
            return []
        pairs = [(query, self.chunks[i].parent_text) for i in ids]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        # Section boosting: add bonus score to chunks from relevant sections
        if target_sections:
            boosted_scores = []
            for idx, (chunk_id, score) in enumerate(zip(ids, scores)):
                chunk_section = self.chunks[chunk_id].section
                if chunk_section in target_sections:
                    boosted_scores.append((chunk_id, float(score) + SECTION_BOOST_SCORE))
                    # self.logger.debug(
                    #     f"  [SectionBoost] #{chunk_id} ({chunk_section}) "
                    #     f"{score:.3f} → {score + SECTION_BOOST_SCORE:.3f}"
                    # )
                else:
                    boosted_scores.append((chunk_id, float(score)))
            ranked = sorted(boosted_scores, key=lambda x: -x[1])
        else:
            ranked = sorted(zip(ids, [float(s) for s in scores]), key=lambda x: -x[1])

        return ranked[:top_k]

    def select_dynamic_k(self, reranked: list, max_k: int) -> list:
        """
        Aggressively select evidence count based on reranker score distribution.
        Key insight: the score formula divides by K — extra low-quality chunks hurt.
        """
        if not reranked or len(reranked) <= 1:
            return reranked[:1] if reranked else []

        selected = [reranked[0]]  # always keep the best chunk

        for i in range(1, min(len(reranked), max_k)):
            chunk_id, score = reranked[i]
            prev_score = reranked[i - 1][1]
            top_score = reranked[0][1]

            # Stop if score drops below absolute threshold
            if score < RERANKER_SCORE_THRESHOLD:
                self.logger.debug(f"  [DynK] Stop: score {score:.3f} < threshold {RERANKER_SCORE_THRESHOLD}")
                break

            # Stop if gap from previous chunk is too large
            if prev_score - score > RERANKER_GAP_THRESHOLD:
                self.logger.debug(f"  [DynK] Stop: gap {prev_score - score:.3f} > {RERANKER_GAP_THRESHOLD}")
                break

            # Stop if score is less than 30% of top score (relative threshold)
            if top_score > 0 and score < top_score * 0.3:
                self.logger.debug(f"  [DynK] Stop: score {score:.3f} < 30% of top {top_score:.3f}")
                break

            selected.append(reranked[i])

        self.logger.debug(f"[DynamicK] max_k={max_k}, selected={len(selected)}")
        return selected

    def retrieve(self, question: str, variants: list, final_k: int,
                 hyde_text: str = "", target_sections: set = None) -> tuple:
        """Full pipeline: multi-query dense+BM25 (+HyDE) → RRF → section-boosted rerank → dynamic-K."""
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

        # HyDE
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

        # self.logger.debug(f"[RRF] dense_unique={dense_unique}, bm25_unique={bm25_unique}, fused={len(fused)}")

        # Stage 3: Cross-encoder reranking with section boosting
        pool_k = min(RERANK_POOL, len(fused))
        reranked = self.rerank(question, fused, pool_k, target_sections=target_sections)

        # Stage 4: Aggressive dynamic K selection
        selected = self.select_dynamic_k(reranked, final_k)

        self.logger.debug(f"[Rerank] pool={pool_k}, selected={len(selected)}:")
        for chunk_id, score in selected:
            c = self.chunks[chunk_id]
            self.logger.debug(f"  #{chunk_id:>3}  score={score:.4f}  [{c.section}]  {c.text[:80]!r}")

        debug = {
            "dense_unique": dense_unique,
            "bm25_unique": bm25_unique,
            "rrf_candidates": len(fused),
            "rerank_pool": pool_k,
            "rerank_scores": [(i, round(s, 4)) for i, s in reranked],
            "final_k": len(selected),
            "target_sections": list(target_sections) if target_sections else [],
        }
        return [self.chunks[i] for i, _ in selected], debug

    def clear(self):
        self.chunks, self.index, self.bm25 = [], None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Generator — Llama-3.2-3B via OpenRouter with CoT Prompt
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
        """Returns (answer_str, debug_info). Uses few-shot CoT prompt optimized for Llama-3.2-3B."""

        # ── Few-Shot CoT System Prompt ──
        system = (
            "You are a precise scientific paper QA system.\n\n"
            "RULES:\n"
            "1. Answer ONLY from the evidence passages. Never use outside knowledge.\n"
            "2. Use the EXACT words, phrases, names, and numbers from the evidence — do not paraphrase.\n"
            "3. Be concise: 1-3 sentences max.\n"
            "4. If the evidence lists multiple items, include ALL of them.\n"
            "5. Do NOT start with filler like \"Based on the evidence\" or \"According to\".\n\n"
            "EXAMPLES:\n\n"
            "Q: Where does the ancient Chinese dataset come from?\n"
            "Evidence: [1] To build the large ancient-modern Chinese dataset, we collected 1.7K bilingual ancient-modern Chinese articles from the internet. More specifically, a large part of the ancient Chinese data we used come from ancient Chinese history records in several dynasties (about 1000BC-200BC) and articles written by celebrities of that era.\n"
            "Reasoning: Passage [1] directly states the source. I extract the exact phrase.\n"
            "Answer: ancient Chinese history records in several dynasties (about 1000BC-200BC) and articles written by celebrities of that era\n\n"
            "Q: What is the BLEU score of the proposed model on the WMT14 En-De test set?\n"
            "Evidence: [1] Our model achieves 28.4 BLEU on WMT14 English-to-German and 41.0 BLEU on WMT14 English-to-French, surpassing all previously published models.\n"
            "Reasoning: The question asks about En-De. Passage [1] states 28.4 BLEU for WMT14 English-to-German.\n"
            "Answer: 28.4 BLEU\n\n"
            "Q: What datasets are used for evaluation?\n"
            "Evidence: [1] We evaluate our approach on three benchmark datasets: SQuAD v1.1, TriviaQA, and Natural Questions (NQ). [2] Additionally, we report results on the MRQA shared task for out-of-domain generalization.\n"
            "Reasoning: Passages [1] and [2] together list all evaluation datasets.\n"
            "Answer: SQuAD v1.1, TriviaQA, Natural Questions (NQ), and the MRQA shared task\n\n"
            "Q: How does the proposed method improve over the baseline?\n"
            "Evidence: [1] Our method reduces training time by 40% while maintaining comparable accuracy. The key innovation is replacing the attention mechanism with a linear projection layer, which reduces the computational complexity from O(n^2) to O(n).\n"
            "Reasoning: Passage [1] explains both the improvement (40% training time reduction) and the mechanism.\n"
            "Answer: It reduces training time by 40% by replacing the attention mechanism with a linear projection layer, reducing computational complexity from O(n^2) to O(n)."
        )

        evidence = chunks[:MAX_EVIDENCE_FOR_LLM]
        ev_lines = []
        total_ev_chars = 0
        for i, c in enumerate(evidence, 1):
            pt = c.parent_text[:PARENT_MAX_CHARS]
            section_tag = f" ({c.section})" if c.section and c.section != "unknown" else ""
            ev_lines.append(f"[{i}]{section_tag} {pt}")
            total_ev_chars += len(pt)

        user = (
            f"Paper: {title}\n\n"
            f"Evidence passages:\n" + "\n\n".join(ev_lines) + "\n\n"
            f"Q: {question}\n"
            f"Reasoning:"
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
                raw_output = (resp.choices[0].message.content or "").strip()
                answer = raw_output

                reasoning_present = False
                reasoning_match = re.search(
                    r"\bReasoning:\s*(.*?)(?:\bAnswer:\s*|$)",
                    raw_output,
                    re.IGNORECASE | re.DOTALL,
                )
                if reasoning_match and reasoning_match.group(1).strip():
                    reasoning_present = True
                else:
                    answer_split = re.split(r"\bAnswer:\s*", raw_output, maxsplit=1, flags=re.IGNORECASE)
                    if len(answer_split) == 2 and answer_split[0].strip():
                        reasoning_present = True

                # Post-process: extract answer after "Answer:" from CoT output
                # Try multiple patterns to be robust
                answer_match = re.search(r'\bAnswer:\s*(.+)', answer, re.IGNORECASE | re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1).strip()
                    # If answer still contains reasoning artifacts, take first paragraph
                    if '\n\n' in answer:
                        answer = answer.split('\n\n')[0].strip()
                # Remove trailing "Reasoning:" artifacts if model repeated format
                answer = re.sub(r'\s*Reasoning:.*$', '', answer, flags=re.DOTALL).strip()

                if resp.usage:
                    debug["actual_prompt_tokens"] = resp.usage.prompt_tokens
                    debug["completion_tokens"] = resp.usage.completion_tokens
                    self.logger.debug(f"[LLM] actual: {resp.usage.prompt_tokens} prompt + "
                                      f"{resp.usage.completion_tokens} completion tokens")
                self.logger.debug(f"[LLM] reasoning_present: {reasoning_present}")
                self.logger.debug(f"[LLM] raw_output=\n{raw_output}")
                self.logger.debug(f"[LLM] parsed_answer=\n{answer}")
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
    def __init__(
        self,
        api_key: str | None,
        logger: logging.Logger,
        enable_hyde: bool = True,
        enable_generation: bool = True,
    ):
        self.logger = logger
        self.enable_hyde = enable_hyde
        self.enable_generation = enable_generation
        self.processor = DocumentProcessor(logger)
        self.retriever = Retriever(logger)
        self.generator = None
        if self.enable_hyde or self.enable_generation:
            self.generator = Generator(api_key=api_key or "ollama", logger=logger)

    @staticmethod
    def determine_k(question: str) -> int:
        """Determine max evidence count. Conservative: fewer is better for ROUGE-L score."""
        q = question.lower()
        list_kw = [
            "list", "what are the", "name the", "how many",
            "what methods", "what techniques", "what datasets",
            "what approaches", "what models", "what features",
            "what types", "what kinds", "what categories",
        ]
        # List-type questions may need more evidence, but still conservative
        return 3 if any(k in q for k in list_kw) else DEFAULT_FINAL_K

    def process_paper(self, entry: dict) -> QAResult:
        title = entry["title"]
        full_text = entry["full_text"]
        question = entry["question"]

        self.logger.debug(f"\n{'='*70}")
        self.logger.debug(f"Paper: {title}")
        self.logger.debug(f"Question: {question}")
        self.logger.debug(f"Text length: {len(full_text):,} chars")
        t_start = time.time()

        # Stage 1: Section-aware chunking
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
        # self.logger.debug(f"[Indexing] done in {dt_index:.2f}s")

        # Stage 3: Query Routing — classify question to target sections
        target_sections = classify_question_sections(question)
        # self.logger.debug(f"[QueryRoute] target_sections={target_sections}")

        # Stage 4: Query Variants
        variants = generate_query_variants(question)
        self.logger.debug(f"[Query variants] ({len(variants)}):")
        for i, v in enumerate(variants):
            self.logger.debug(f"  Q{i}: {v}")

        # Stage 5: HyDE
        t0 = time.time()
        hyde_text = ""
        if self.enable_hyde and self.generator is not None:
            hyde_text = self.generator.generate_hyde(title, question)
            dt_hyde = time.time() - t0
            self.logger.debug(f"[HyDE] generated in {dt_hyde:.2f}s")
        else:
            dt_hyde = 0.0
            self.logger.debug("[HyDE] disabled")

        # Stage 6: Retrieval with section boosting
        k = self.determine_k(question)
        t0 = time.time()
        retrieved, retrieval_debug = self.retriever.retrieve(
            question, variants, k,
            hyde_text=hyde_text,
            target_sections=target_sections,
        )
        dt_retrieval = time.time() - t0
        self.logger.debug(f"[Retrieval] K={k}, got {len(retrieved)} chunks in {dt_retrieval:.2f}s")

        # Stage 7: LLM Generation
        t0 = time.time()
        if self.enable_generation and self.generator is not None:
            answer, gen_debug = self.generator.generate(title, question, retrieved)
            dt_llm = time.time() - t0
        else:
            answer, gen_debug = "", {}
            dt_llm = 0.0
            self.logger.debug("[LLM] generation disabled")
        # self.logger.debug(f"[LLM] generated in {dt_llm:.2f}s")

        # Stage 8: Cleanup
        self.retriever.clear()

        total_time = time.time() - t_start
        evidence = [c.text for c in retrieved]

        self.logger.debug(f"[Result] answer ({len(answer)} chars): {answer}")
        self.logger.debug(f"[Result] evidence: {len(evidence)} chunks, sections={[c.section for c in retrieved]}")
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
            ans_preview = result.answer[:100].replace("\n", " ")
            # self.logger.debug(f"  → {ans_preview}")

        elapsed = time.time() - t_all
        n = len(dataset)
        # tqdm.write(f"\nDone: {n} papers in {elapsed:.1f}s ({elapsed/n:.1f}s/paper avg)")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator — local ROUGE-L check (matches score_public.py formula exactly)
# ═══════════════════════════════════════════════════════════════════════════════
class Evaluator:
    def __init__(self, logger=None):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.logger = logger

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

        def log_debug(message: str):
            if self.logger:
                self.logger.debug(message)

        log_debug(f"\n{'='*70}")
        log_debug("  Evidence Score Evaluation (ROUGE-L, same formula as TA)")
        log_debug(f"{'='*70}")

        tqdm.write(f"\n{'='*70}")
        tqdm.write("  Evidence Score Evaluation (ROUGE-L, same formula as TA)")
        tqdm.write(f"{'='*70}")

        for r in results:
            gt = gt_map.get(r.title)
            if gt is None:
                log_debug(f"  [SKIP] {r.title[:50]} — not in ground truth")
                continue
            golden_ev = gt.get("evidence", [])
            s = self.evidence_score_single(r.evidence, golden_ev)
            ev_scores.append(s)
            log_debug(f"  {s:.4f}  K={len(r.evidence):>2}  gt_ev={len(golden_ev)}  title={r.title[:50]}")

        if not ev_scores:
            log_debug("  No papers scored.")
            return {"mean_evidence_score": 0, "n": 0}

        avg = sum(ev_scores) / len(ev_scores)
        log_debug(f"\n  {'─'*60}")
        log_debug(f"  Mean Evidence Score : {avg:.5f}  (n={len(ev_scores)})")
        log_debug(f"  {'─'*60}")

        tqdm.write(f"  Mean Evidence Score : {avg:.5f}  (n={len(ev_scores)})")

        # K distribution stats
        k_values = [len(r.evidence) for r in results if gt_map.get(r.title)]
        if k_values:
            avg_k = sum(k_values) / len(k_values)
            log_debug(f"  Avg K={avg_k:.1f}, min={min(k_values)}, max={max(k_values)}")

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

    logger, log_path = setup_logging(script_dir / "logs")

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

    api_key = LLM_API_KEY
    if LLM_PROVIDER == "openrouter" and not api_key:
        tqdm.write("ERROR: Set OPENROUTER_API_KEY or LLM_API_KEY environment variable.")
        sys.exit(1)

    tqdm.write(f"{'#'*70}")
    tqdm.write(f"# HW2 RAG Pipeline v2 (Optimized) — Student {STUDENT_ID}")
    tqdm.write(f"{'#'*70}")
    tqdm.write(f"  Dataset  : {dataset_path}")
    tqdm.write(f"  Output   : {output_path}")
    tqdm.write(f"  Log file : {log_path}")
    tqdm.write(f"  Eval mode: {args.eval}  (sample={args.sample if args.sample > 0 else 'all'})")
    tqdm.write(f"  LLM src  : {LLM_PROVIDER} @ {LLM_BASE_URL}")
    tqdm.write(f"  Embed    : {EMBED_MODEL}")
    tqdm.write(f"  Reranker : {RERANKER_MODEL}")
    tqdm.write(f"  LLM      : {LLM_MODEL}")
    tqdm.write(f"  Chunk    : target={CHILD_TARGET_CHARS}, max={CHILD_MAX_CHARS} chars, overlap={CHUNK_SENTENCE_OVERLAP}")
    tqdm.write(f"  Parent   : window={PARENT_WINDOW_CHUNKS}, max={PARENT_MAX_CHARS} chars")
    tqdm.write(f"  Retrieval: dense_k={DENSE_TOP_K}, bm25_k={BM25_TOP_K}, "
               f"rrf_k={RRF_TOP_K}, final_k={DEFAULT_FINAL_K}")
    tqdm.write(f"  DynK     : score_thresh={RERANKER_SCORE_THRESHOLD}, gap_thresh={RERANKER_GAP_THRESHOLD}")
    tqdm.write(f"  Section  : boost={SECTION_BOOST_SCORE}")
    tqdm.write(f"  LLM ctx  : max {MAX_EVIDENCE_FOR_LLM} parent chunks, {PARENT_MAX_CHARS} chars each")
    tqdm.write("")

    logger.debug(f"Config: embed={EMBED_MODEL}, reranker={RERANKER_MODEL}, llm={LLM_MODEL}")
    logger.debug(f"LLM transport: provider={LLM_PROVIDER}, base_url={LLM_BASE_URL}")
    logger.debug(f"Chunk: target={CHILD_TARGET_CHARS}, max={CHILD_MAX_CHARS}, "
                 f"parent_window={PARENT_WINDOW_CHUNKS}, parent_max={PARENT_MAX_CHARS}, "
                 f"sentence_overlap={CHUNK_SENTENCE_OVERLAP}")
    logger.debug(f"DynK: score_thresh={RERANKER_SCORE_THRESHOLD}, gap_thresh={RERANKER_GAP_THRESHOLD}")

    if not dataset_path.exists():
        tqdm.write(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path) as f:
        dataset = json.load(f)
    tqdm.write(f"Loaded {len(dataset)} papers from {dataset_path.name}")

    if args.eval and args.sample > 0 and args.sample < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.sample)
        tqdm.write(f"Sampled {len(dataset)} papers (seed={args.seed})")
    tqdm.write("")

    pipeline = RAGPipeline(api_key=api_key, logger=logger)
    results = pipeline.run(dataset)

    if not args.eval:
        output = [{"title": r.title, "answer": r.answer, "evidence": r.evidence} for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        tqdm.write(f"\nSaved {len(output)} results → {output_path}")

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

    if args.eval:
        evaluator = Evaluator(logger=logger)
        evaluator.evaluate(results, dataset)

    tqdm.write(f"\nLog saved → {log_path}")


if __name__ == "__main__":
    main()
