#!/usr/bin/env python3
"""
HW2: Document QA based on RAG


Default LLM provider: openrouter
For local Ollama, prefix commands with LLM_PROVIDER=ollama

Usage:
  uv run python 111511157.py                                                    # Process private_dataset.json
  uv run python 111511157.py --eval                                             # Evaluate on public_dataset.json (writes outputs/public.json)
  uv run python 111511157.py --eval --sample 0                                  # Evaluate all papers (writes outputs/public.json)
    
    --output but forces it into outputs/ folder

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
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
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
CHILD_MAX_CHARS = 350
PARENT_WINDOW_CHUNKS = 2
CHUNK_SENTENCE_OVERLAP = 2  # NEW: 1-sentence overlap between consecutive chunks
PARENT_MAX_CHARS = 1200       # REVERTED: 800 truncated golden evidence text

# Retrieval
DENSE_TOP_K = 60              # CHANGED: was 40, wider net catches more candidates
BM25_TOP_K = 60              # CHANGED: was 40
RRF_TOP_K = 30               # CHANGED: was 20, more candidates for reranking
RRF_K = 60
DEFAULT_FINAL_K = 1
RERANK_POOL = 30             # CHANGED: was 20, deeper reranking pool

# Dynamic K selection — TIGHTENED significantly
RERANKER_SCORE_THRESHOLD = 0.0    # CHANGED: was -1.0 — only keep positively-scored chunks
RERANKER_GAP_THRESHOLD = 2.0      # CHANGED: was 5.0 — drop if score gap is large

# Section boosting
SECTION_BOOST_SCORE = 1.5  # NEW: bonus added to reranker score for matching sections

# Embedding query prefix (model-specific — must match embedding model's training format)
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge-*
EMBED_PASSAGE_PREFIX = ""  # Snowflake arctic-embed: no passage prefix needed

# LLM
MAX_EVIDENCE_FOR_LLM = 5
LLM_CONTEXT_K = 5            # CHANGED: was 20, reduced — 3B model works best with focused context
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 256
SC_VOTES = 1                 # CHANGED: was 3, disabled — didn't prove a win, triples API calls
SC_TEMPERATURE = 0.3         # Temperature for voting runs (0.0 for first, SC_TEMP for rest)

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

    def refine_evidence(self, question: str, reranked_pool: list, full_text: str,
                        max_k: int = 1, logger=None) -> list[str]:
        """
        Sentence-level evidence refinement for higher ROUGE-L.

        Instead of submitting whole chunks (~200 chars with arbitrary boundaries),
        extract individual sentences and multi-sentence windows from the top reranked
        chunks + keyword-matched sentences, re-score them with the cross-encoder,
        and return the best non-overlapping spans.

        This dramatically improves ROUGE-L because golden evidence is typically
        1-3 exact sentences from the paper.
        """
        candidates = []
        seen_keys = set()

        def _clean_evidence_text(text):
            """Light-clean evidence spans. Preserve BIBREF/TABREF/INLINEFORM markers
            because ~35% of golden evidences contain them — stripping hurts ROUGE-L."""
            # Remove section header prefixes like "Abstract\n", "Conclusions\n", "Datasets :::\n"
            text = re.sub(r'^(?:Abstract|Introduction|Conclusions?|Related Work|Discussion|'
                          r'Results?|Methods?|Experiments?|Background|Evaluation)\s*\n',
                          '', text, flags=re.IGNORECASE)
            # Remove "Section ::: Subsection\n" style headers
            text = re.sub(r'^[A-Z][^\n]{0,60}(?:\s*:::)+[^\n]*\n', '', text)
            # NOTE: Do NOT strip BIBREF/TABREF/FIGREF/INLINEFORM/DISPLAYFORM here —
            # golden evidence often contains them (verified: 35% of golden has BIBREF).
            return text.strip()

        def _add(text):
            text = _clean_evidence_text(text)
            if len(text) < 15 or len(text) > 800:
                return
            key = text[:60].lower()
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append(text)

        # Source 1: Sentences + sliding windows from top reranked chunks' parent text
        for chunk in reranked_pool[:20]:  # CHANGED: was 15 → 20, deeper pool
            # Also add the raw child chunk text as a candidate
            _add(chunk.text)
            try:
                sents = nltk.sent_tokenize(chunk.parent_text)
            except Exception:
                sents = chunk.parent_text.split('. ')
            for i, sent in enumerate(sents):
                _add(sent)
                # 2-sentence window (golden evidence often spans 2 sentences)
                if i + 1 < len(sents):
                    _add(sent.strip() + ' ' + sents[i + 1].strip())
                # 3-sentence window
                if i + 2 < len(sents):
                    _add(' '.join(s.strip() for s in sents[i:i + 3]))

        # Source 2: Full-text sentence search — catches evidence that retrieval missed entirely
        # Golden evidence can be ANY sentence in the paper; chunk-based retrieval misses ~50% of them
        try:
            all_sents = nltk.sent_tokenize(full_text)
        except Exception:
            all_sents = full_text.split('. ')

        # Score all sentences by keyword overlap with question
        q_keywords = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', question)
                        if w.lower() not in {'the', 'this', 'that', 'what', 'which', 'how',
                                            'does', 'did', 'are', 'was', 'were', 'been',
                                            'have', 'has', 'had', 'for', 'with', 'from',
                                            'they', 'their', 'there', 'used', 'using',
                                            'paper'})
        # Adaptive threshold: with few keywords, single match is enough; otherwise require 2
        kw_threshold = 1 if len(q_keywords) <= 3 else 2
        for i, sent in enumerate(all_sents):
            if len(sent) < 20 or len(sent) > 500:
                continue
            sent_lower = sent.lower()
            hit_count = sum(1 for kw in q_keywords if kw in sent_lower)
            if hit_count >= kw_threshold:
                _add(sent)
                # 2-sentence window
                if i + 1 < len(all_sents):
                    _add(sent.strip() + ' ' + all_sents[i + 1].strip())
                # 3-sentence window for higher recall
                if i + 2 < len(all_sents) and hit_count >= kw_threshold:
                    _add(' '.join(s.strip() for s in all_sents[i:i + 3]))

        # Source 3: Keyword-matched sentences (legacy, may overlap with source 2 — dedup handles it)
        kw_sents = extract_keyword_sentences(full_text, question, max_sentences=20)
        for sent in kw_sents:
            _add(sent)

        if not candidates:
            return [reranked_pool[0].text] if reranked_pool else [""]

        # Score ALL candidates with cross-encoder reranker (one batch call)
        pairs = [(question, c) for c in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        scored = sorted(
            zip(candidates, [float(s) for s in scores]),
            key=lambda x: -x[1]
        )

        if logger:
            top_n = min(5, len(scored))
            logger.debug(f"[RefineEvidence] {len(candidates)} candidates, "
                         f"top scores: {[f'{s:.3f}' for _, s in scored[:top_n]]}")

        # Select non-overlapping top spans with quality gating
        selected = []
        top_score = scored[0][1] if scored else 0

        for text, score in scored:
            # Quality gate: stricter threshold for 2nd+ evidence piece
            # (scoring formula divides by K — filler evidence kills score)
            if selected:
                if score < 0.5:  # Strict: 2nd evidence must be confidently relevant
                    break
                if top_score > 0 and score < top_score * 0.4:
                    break

            # Overlap check: skip if too similar to already-selected spans
            overlap = False
            for s in selected:
                if text in s or s in text:
                    overlap = True
                    break
                w1 = set(text.lower().split())
                w2 = set(s.lower().split())
                denom = min(len(w1), len(w2))
                if denom > 0 and len(w1 & w2) / denom > 0.6:
                    overlap = True
                    break

            if not overlap:
                selected.append(text)
            if len(selected) >= max_k:
                break

        if logger:
            logger.debug(f"[RefineEvidence] selected {len(selected)} evidence spans")
            for i, s in enumerate(selected):
                logger.debug(f"  EV[{i}]: {s[:150]}")

        return selected if selected else [reranked_pool[0].text]

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
            "reranked_pool": [self.chunks[i] for i, _ in reranked],  # full reranked pool for LLM context
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

    # ── Question Type Classification ──
    @staticmethod
    def classify_question_type(question: str) -> str:
        q = question.lower().strip()
        # Check question-word prefix first to avoid false yesno on "How is the..."
        starts_with_qword = any(q.startswith(w) for w in [
            "what ", "how ", "why ", "where ", "when ", "who ", "which ",
            "in what", "to what", "for what",
        ])
        if any(kw in q for kw in ["how many", "how much", "what is the size",
                                   "what is the number", "how long", "how small",
                                   "how large", "how big", "how often"]):
            return "number"
        if any(kw in q for kw in ["what are the", "what were the", "what methods",
                                   "what techniques", "what datasets", "what approaches",
                                   "what models", "what features", "what types",
                                   "which", "what baselines", "what existing",
                                   "what languages", "what two", "what three",
                                   "list", "name the", "what metrics",
                                   "what corpora", "what tasks"]):
            return "list"
        # yesno ONLY if question starts with is/are/does/do/can/will etc.
        if not starts_with_qword and any(kw in q for kw in [
                "is there", "are there", "does the", "do the",
                "is the", "can the", "is it", "does it",
                "do they", "are the", "can it", "will the"]):
            return "yesno"
        if any(kw in q for kw in ["why", "what is the reason", "what causes"]):
            return "reason"
        if any(kw in q for kw in ["how do", "how does", "how is", "how are", "how to"]):
            return "method"
        return "factoid"

    # ── Span Snapping: map LLM output back to exact evidence text ──
    @staticmethod
    def snap_to_evidence_span(answer: str, chunks: list) -> str:
        """If the LLM paraphrased, find the closest matching span in evidence
        and replace with the verbatim text. This increases extractive fidelity."""
        from nltk import sent_tokenize

        if not answer or not chunks or len(answer) < 5:
            return answer

        # Handle yesno: strip prefix, snap the rest, prepend back
        yesno_prefix = ""
        yesno_match = re.match(r'^(Yes|No)[.,;:!\s]+', answer, re.IGNORECASE)
        if yesno_match:
            yesno_prefix = yesno_match.group(0)
            answer_body = answer[len(yesno_prefix):].strip()
            if answer_body:
                answer = answer_body
            else:
                return answer  # just "Yes" or "No"

        answer_words = set(w.lower() for w in re.findall(r'\b\w{2,}\b', answer))
        if len(answer_words) < 2:
            return yesno_prefix + answer if yesno_prefix else answer

        best_span = None
        best_score = 0.0

        for c in chunks:
            text = c.parent_text
            try:
                sentences = sent_tokenize(text)
            except Exception:
                sentences = text.split('. ')
            if not sentences:
                continue

            # Try 1-, 2-, and 3-sentence windows
            for window in range(1, min(4, len(sentences) + 1)):
                for i in range(len(sentences) - window + 1):
                    span = " ".join(sentences[i:i + window])
                    span_words = set(w.lower() for w in re.findall(r'\b\w{2,}\b', span))
                    if not span_words:
                        continue

                    overlap = answer_words & span_words
                    if not overlap:
                        continue
                    # Recall: how much of the answer is covered
                    recall = len(overlap) / len(answer_words)
                    # Precision: how focused is the span
                    precision = len(overlap) / len(span_words)
                    # F1
                    f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

                    # Prefer shorter spans at equal F1 (more precise)
                    if f1 > best_score or (f1 == best_score and best_span and len(span) < len(best_span)):
                        best_score = f1
                        best_span = span

        # Snap if reasonable match — threshold balances extractive fidelity vs wrong snaps.
        # Guard: don't snap to a span that's much shorter than the answer unless score is
        # very high (otherwise we collapse "KAR is an end-to-end MRC model" into bare "KAR").
        if best_score >= 0.4 and best_span:
            orig_len = len(answer)
            span_len = len(best_span)
            if span_len < 0.5 * orig_len and best_score < 0.7:
                return yesno_prefix + answer if yesno_prefix else answer
            return yesno_prefix + best_span if yesno_prefix else best_span
        return yesno_prefix + answer if yesno_prefix else answer

    # ── Answer Post-Processing ──
    @staticmethod
    def clean_answer(answer: str, chunks: list) -> str:
        """Strip meta-commentary, handle IDK, normalize answer."""
        # 0. Remove "Here are..." / "Here is..." meta-commentary headers
        #    These are the #1 source of 0-score answers (22/100 in latest run)
        answer = re.sub(
            r"^Here (are|is) .{0,150}?[:\n]+\s*",
            "", answer, flags=re.IGNORECASE
        )
        # Also: "The answer is:", "The answer to the question is:"
        answer = re.sub(
            r"^The answer( to .{0,80}?)?\s*(is|would be)[:\s]+",
            "", answer, flags=re.IGNORECASE
        )
        # 0b. If stripping left us empty, fallback to top chunk sentence
        if not answer.strip() and chunks:
            from nltk import sent_tokenize
            sentences = sent_tokenize(chunks[0].parent_text)
            if sentences:
                answer = sentences[0]
        # 1. Remove meta-commentary prefixes
        answer = re.sub(
            r"^(Based on|According to|From|As (stated|mentioned|described|shown) in|"
            r"The (evidence|passage|paper|text|context) (states|mentions|shows|indicates|describes|suggests) that)"
            r"\s*(the\s+)?(evidence|passage|paper|text|context)?[^,]*,\s*",
            "", answer, flags=re.IGNORECASE
        )
        # 2. Remove "Passage [N] states that..." references
        answer = re.sub(
            r"Passage[s]?\s*\[\d+\](\s*(and|,)\s*(\[\d+\]|Passage\s*\[\d+\]))*\s*"
            r"(both\s+)?(state|mention|show|indicate|describe|explain|note|say|suggest)s?\s+that\s+",
            "", answer, flags=re.IGNORECASE
        )
        answer = re.sub(r"\bPassage[s]?\s*\[\d+\](\s*(and|,)\s*\[\d+\])*\s*", "", answer)
        # 3. IDK fallback — replace with first sentence of top chunk
        # Strip "None." / "N/A" bare prefixes first (LLM sometimes emits literal "None.")
        answer = re.sub(r'^(None|N/A|NA|No answer|No\.)\s*[.\-:]?\s*', '', answer, flags=re.IGNORECASE)
        idk_patterns = [
            r"none of the .*(passage|evidence)",
            r"^none\s*[.\-:]",
            r"not (explicitly|specifically|directly) mentioned",
            r"does(n'?t| not) (mention|contain|include|provide|say|have)",
            r"no (information|mention|evidence|specific|direct)",
            r"not (answered|specified) (in|by)",
            r"is not answered",
            r"no (direct|explicit) (mention|answer)",
            r"unable to (find|determine|answer)",
            r"cannot (find|determine|answer)",
            r"not (enough|sufficient)",
            r"i (don'?t|do not|cannot) know",
        ]
        if any(re.search(p, answer, re.IGNORECASE) for p in idk_patterns) and chunks:
            from nltk import sent_tokenize
            sentences = sent_tokenize(chunks[0].parent_text)
            if sentences:
                answer = sentences[0]
        # 4. If answer is just a table/figure reference, extract surrounding context
        if chunks and re.match(r"^(Table|Figure|Fig\.)\s*(TABREF|FIGREF)?\d+\.?$", answer.strip(), re.IGNORECASE):
            from nltk import sent_tokenize
            sentences = sent_tokenize(chunks[0].parent_text)
            if sentences:
                answer = sentences[0]
        # 5. Remove trailing explanatory clauses
        answer = re.sub(r"\s*,?\s*as (evidenced|shown|mentioned|stated|described) (by|in).*$",
                        "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"\s*,?\s*which (lists?|shows?|indicates?|suggests?|demonstrates?).*$",
                        "", answer, flags=re.IGNORECASE)
        # 6. Remove duplicate sentences (sentence-level dedup, not comma-splitting)
        from nltk import sent_tokenize
        try:
            sents = sent_tokenize(answer)
        except Exception:
            sents = [answer]
        if len(sents) >= 2:
            seen_normalized = []
            deduped = []
            for s in sents:
                norm = re.sub(r'[.\s]+$', '', s.strip()).lower()
                if norm and norm not in seen_normalized:
                    seen_normalized.append(norm)
                    deduped.append(s.strip())
            if deduped:
                answer = " ".join(deduped)
        # 7. Remove paper reference markers (noise in extracted answers)
        answer = re.sub(r'\s*BIBREF\d+', '', answer)
        answer = re.sub(r'(?:Section\s+)?SECREF\d+', '', answer)
        answer = re.sub(r'(?:Table\s+)?TABREF\d+', 'the table', answer)
        answer = re.sub(r'(?:Figure\s+)?FIGREF\d+', 'the figure', answer)
        answer = re.sub(r'INLINEFORM\d+', '', answer)
        answer = re.sub(r'DISPLAYFORM\d+', '', answer)
        answer = re.sub(r'(,\s*)+', ', ', answer)
        answer = re.sub(r'\(\s*,?\s*\)', '', answer)
        answer = re.sub(r'§\s*', '', answer)
        # 8. Fix number formatting: "40, 000" → "40,000", "19, 300" → "19,300"
        answer = re.sub(r'(\d),\s+(\d{3})', r'\1,\2', answer)
        # 9. Clean up
        answer = answer.strip().strip('"').strip("'").strip()
        answer = re.sub(r'^[,\s]+|[,\s]+$', '', answer)
        return answer

    def _build_llm_context(self, chunks: list, extra_chunks: list = None,
                           extra_sentences: list = None) -> tuple:
        """Build evidence lines and supplementary context for LLM prompt.
        Returns (ev_lines, supp_block, total_ev_chars)."""
        evidence = chunks[:MAX_EVIDENCE_FOR_LLM]
        ev_lines = []
        total_ev_chars = 0
        for i, c in enumerate(evidence, 1):
            text = c.parent_text[:PARENT_MAX_CHARS]
            ev_lines.append(f"[{i}] {text}")
            total_ev_chars += len(text)

        # Supplementary context from extra reranked chunks + keyword sentences
        supp_lines = []
        seen_texts = {c.parent_text[:100] for c in evidence}

        if extra_chunks:
            for c in extra_chunks[:LLM_CONTEXT_K - len(evidence)]:
                key = c.parent_text[:100]
                if key not in seen_texts:
                    seen_texts.add(key)
                    supp_lines.append(c.parent_text[:PARENT_MAX_CHARS])

        if extra_sentences:
            for sent in extra_sentences:
                key = sent[:100]
                if key not in seen_texts:
                    seen_texts.add(key)
                    supp_lines.append(sent)

        supp_block = ""
        if supp_lines:
            supp_block = "\n\nAdditional context from the paper:\n" + "\n".join(
                f"- {s}" for s in supp_lines[:8])

        return ev_lines, supp_block, total_ev_chars

    def _parse_raw_answer(self, raw_output: str, chunks: list) -> str:
        """Parse and clean a single LLM raw output into a clean answer."""
        answer = raw_output
        # Extract answer part if model outputs "Answer:" or "Reasoning:"
        answer_match = re.search(r'\bAnswer:\s*(.+)', answer, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        answer = re.sub(r'\s*Reasoning:.*$', '', answer, flags=re.DOTALL).strip()
        if '\n\n' in answer:
            answer = answer.split('\n\n')[0].strip()
        answer = self.clean_answer(answer, chunks)
        return answer

    def _score_extractiveness(self, answer: str, chunks: list) -> float:
        """Score how extractive an answer is (word overlap with evidence text)."""
        if not answer or not chunks:
            return 0.0
        answer_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', answer))
        if not answer_words:
            return 0.0
        evidence_words = set()
        for c in chunks[:MAX_EVIDENCE_FOR_LLM]:
            evidence_words.update(w.lower() for w in re.findall(r'\b\w{3,}\b', c.parent_text))
        if not evidence_words:
            return 0.0
        return len(answer_words & evidence_words) / len(answer_words)

    def generate(self, title: str, question: str, chunks: list,
                 extra_chunks: list = None, extra_sentences: list = None,
                 max_retries: int = 5) -> tuple:
        """Returns (answer_str, debug_info). Self-consistency voting for Llama-3.2-3B."""

        # ── Question Type ──
        q_type = self.classify_question_type(question)

        # ── System Prompt: Detailed with examples (3B needs examples) ──
        type_instructions = {
            "number": "Find the EXACT number/quantity in the text. Copy the full phrase including the number with its unit and surrounding noun. Example: '$0.3$ million records' not just '0.3 million'. Example: 'roughly 40,000 Manhattan listings' not just '40,000'.",
            "list": "List ALL specific items by their exact names from the text, separated by commas. Example: 'Level A: Offensive language Detection, Level B: Categorization of Offensive Language, Level C: Offensive Language Target Identification'. Example: 'BiLSTM, BiLSTM+CNN, BiLSTM+CRF, BiLSTM+CNN+CRF, CNN, Stanford CRF'.",
            "yesno": "Start with Yes or No, then copy the sentence from the text that supports your answer.",
            "reason": "Copy the sentence from the text that explains the reason. Do NOT paraphrase.",
            "method": "Copy the sentence(s) from the text that describe the specific method, technique, or process.",
            "factoid": "Copy the specific phrase or sentence from the text that directly answers the question. Be precise — include the specific names, numbers, or details asked about.",
        }
        type_hint = type_instructions.get(q_type, type_instructions["factoid"])

        system = (
            "You are an extractive QA system for scientific papers. Your job is to find and COPY the exact answer from the text.\n\n"
            "CRITICAL RULES:\n"
            "1. Find the sentence(s) that DIRECTLY answer the question and COPY them verbatim.\n"
            "2. Do NOT copy sentences that merely RESTATE or INTRODUCE the topic. "
            "For example, if the question is 'What is the core component for KBQA?', do NOT answer 'Relation detection is a core component for KBQA.' — "
            "instead find the sentence that describes WHAT that component IS or HOW it works.\n"
            "3. Be SPECIFIC: if the question asks 'what model?', give the model NAME (e.g. 'BERTbase'), not a description like 'a transformer model'.\n"
            "4. If the question asks 'how many/what size?', give the EXACT number with context (e.g. '$0.3$ million records').\n"
            "5. If the question asks for a list, list ALL specific items by name.\n"
            "6. Answer in 1-2 sentences maximum. Do NOT write paragraphs.\n"
            "7. NEVER start with 'Based on', 'According to', 'The paper states', 'The answer is', 'Here are', 'We', 'In this paper'.\n"
            "8. NEVER paraphrase or explain — just copy the relevant text.\n"
            "9. NEVER say 'not mentioned' or 'I don't know'.\n"
            "10. NEVER echo the question back. Start directly with the specific answer content.\n\n"
            f"FORMAT: {type_hint}"
        )

        # ── Build LLM context: focused, high-quality parent chunks ──
        ev_lines, supp_block, total_ev_chars = self._build_llm_context(
            chunks, extra_chunks, extra_sentences)

        user = (
            f"Paper: {title}\n\n"
            f"Evidence:\n" + "\n\n".join(ev_lines) + supp_block + "\n\n"
            f"Question: {question}\n"
            f"Find the specific answer in the evidence above. Copy the exact words.\n"
            f"Answer:"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        prompt_chars = len(system) + len(user)
        est_tokens = prompt_chars // 4

        self.logger.debug(f"[LLM] n_evidence={len(ev_lines)}, ev_chars={total_ev_chars}, "
                          f"prompt_chars={prompt_chars}, est_tokens={est_tokens}")

        debug = {
            "n_evidence_for_llm": len(ev_lines),
            "evidence_total_chars": total_ev_chars,
            "prompt_chars": prompt_chars,
            "est_prompt_tokens": est_tokens,
        }

        # ── Self-Consistency Voting: generate N times, pick most extractive ──
        candidates = []
        temperatures = [LLM_TEMPERATURE] + [SC_TEMPERATURE] * (SC_VOTES - 1)

        for vote_idx, temp in enumerate(temperatures):
            for attempt in range(max_retries):
                try:
                    resp = self.client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=temp,
                        max_tokens=LLM_MAX_TOKENS,
                    )
                    raw_output = (resp.choices[0].message.content or "").strip()
                    answer = self._parse_raw_answer(raw_output, chunks)

                    if vote_idx == 0 and resp.usage:
                        debug["actual_prompt_tokens"] = resp.usage.prompt_tokens
                        debug["completion_tokens"] = resp.usage.completion_tokens

                    self.logger.debug(f"[LLM] vote {vote_idx}: q_type={q_type}, raw={raw_output[:150]}")
                    self.logger.debug(f"[LLM] vote {vote_idx}: parsed={answer[:150]}")
                    candidates.append(answer)
                    break
                except Exception as e:
                    wait = min(2 ** (attempt + 1), 30)
                    self.logger.warning(f"[LLM] Error (vote {vote_idx}, attempt {attempt+1}): {e}")
                    time.sleep(wait)

        if not candidates:
            self.logger.error("[LLM] All votes failed, returning fallback answer.")
            return "Unable to generate answer.", debug

        # Pick the best candidate: highest extractive overlap with evidence
        if len(candidates) == 1:
            best = candidates[0]
        else:
            scored = [(c, self._score_extractiveness(c, chunks)) for c in candidates]
            scored.sort(key=lambda x: -x[1])
            best = scored[0][0]
            self.logger.debug(f"[SC] {len(candidates)} candidates, "
                              f"scores={[f'{s:.2f}' for _, s in scored]}, "
                              f"picked: {best[:100]}")

        # ── Anti-Echo Retry: if answer echoes the question, retry with stronger prompt ──
        if self.is_echo(best, question):
            self.logger.debug(f"[AntiEcho] detected echo, retrying with targeted prompt")
            anti_echo_system = (
                "You are answering a question about a scientific paper. "
                "The previous answer was wrong because it just restated the question instead of answering it.\n\n"
                "IMPORTANT: Do NOT repeat the question. Find the SPECIFIC answer in the text.\n"
                f"- Question type: {q_type}\n"
                f"- {type_hint}\n\n"
                "Look for sentences that contain specific names, numbers, methods, or results — NOT sentences that introduce or describe the topic."
            )
            anti_echo_user = (
                f"Paper: {title}\n\n"
                f"Evidence:\n" + "\n\n".join(ev_lines) + supp_block + "\n\n"
                f"Question: {question}\n"
                f"BAD answer (just restates the question): {best}\n"
                f"Find the REAL answer from the evidence. Copy the exact text.\n"
                f"Answer:"
            )
            try:
                resp2 = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "system", "content": anti_echo_system},
                              {"role": "user", "content": anti_echo_user}],
                    temperature=0.1, max_tokens=LLM_MAX_TOKENS,
                )
                retry_answer = self._parse_raw_answer(
                    (resp2.choices[0].message.content or "").strip(), chunks)
                if retry_answer and not self.is_echo(retry_answer, question) and not self.is_idk(retry_answer):
                    self.logger.debug(f"[AntiEcho] retry succeeded: {retry_answer[:150]}")
                    best = retry_answer
                else:
                    self.logger.debug(f"[AntiEcho] retry didn't help, keeping original")
            except Exception as e:
                self.logger.warning(f"[AntiEcho] retry failed: {e}")

        # Apply span snapping to force extractive answer
        best = self.snap_to_evidence_span(best, chunks)

        # Clean ref markers that snap may have re-introduced from raw chunk text
        best = re.sub(r'\s*BIBREF\d+', '', best)
        best = re.sub(r'(?:Section\s+)?SECREF\d+', '', best)
        best = re.sub(r'(?:Table\s+)?TABREF\d+', 'the table', best)
        best = re.sub(r'(?:Figure\s+)?FIGREF\d+', 'the figure', best)
        best = re.sub(r'INLINEFORM\d+', '', best)
        best = re.sub(r'DISPLAYFORM\d+', '', best)
        best = re.sub(r'\(\s*,?\s*\)', '', best)
        best = re.sub(r'(,\s*)+', ', ', best)
        best = re.sub(r'§\s*', '', best)
        best = best.strip()

        debug["sc_candidates"] = len(candidates)
        return best, debug

    def generate_fallback(self, title: str, question: str, chunks: list,
                          extra_chunks: list = None, extra_sentences: list = None) -> tuple:
        """Fallback generation with a more permissive prompt for IDK recovery."""
        system = (
            "Answer the question using the text below. "
            "Give a short, direct answer. Use the exact words from the text. "
            "Never say 'not mentioned' or 'I don't know'. "
            "NEVER start with 'Here are', 'Based on', or 'The answer is'. "
            "Start directly with the answer content."
        )
        ev_lines, supp_block, _ = self._build_llm_context(
            chunks, extra_chunks, extra_sentences)

        user = (
            f"Paper: {title}\n\n"
            f"Evidence:\n" + "\n\n".join(ev_lines) + supp_block + "\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL, messages=messages,
                temperature=0.3, max_tokens=LLM_MAX_TOKENS,
            )
            answer = self._parse_raw_answer(
                (resp.choices[0].message.content or "").strip(), chunks)
            self.logger.debug(f"[LLM-fallback] answer={answer}")
            return answer, {}
        except Exception as e:
            self.logger.warning(f"[LLM-fallback] Error: {e}")
            return "", {}

    # ── IDK detection ──
    _IDK_PATTERNS = [
        "none of the", "not mentioned", "not explicitly",
        "does not mention", "doesn't mention", "does not contain",
        "no information", "unable to", "cannot find", "not enough",
        "not specifically", "not directly", "no specific",
        "doesn't provide", "does not provide", "not provided",
        "no direct mention", "no direct answer", "is not answered",
        "not answered in", "no explicit", "not specified in",
        "evidence does not", "evidence doesn't",
    ]

    @classmethod
    def is_idk(cls, answer: str) -> bool:
        a = answer.lower()
        return any(p in a for p in cls._IDK_PATTERNS)

    @staticmethod
    def is_echo(answer: str, question: str) -> bool:
        """Detect if the answer is just echoing/restating the question."""
        if not answer or not question:
            return False
        # Normalize
        a_words = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', answer))
        q_words = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', question))
        # Remove common stop words
        stop = {'the', 'and', 'for', 'are', 'was', 'were', 'has', 'have', 'had',
                'this', 'that', 'with', 'from', 'into', 'also', 'more', 'than',
                'their', 'they', 'them', 'some', 'other', 'been', 'not', 'will'}
        a_words -= stop
        q_words -= stop
        if not a_words or not q_words:
            return False
        overlap = len(a_words & q_words)
        # Echo if >55% of answer words come from question
        ratio = overlap / len(a_words)
        return ratio > 0.55 and len(a_words) < 25


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
# Keyword Sentence Extraction — supplementary LLM context from full_text
# ═══════════════════════════════════════════════════════════════════════════════
_KW_STOP = frozenset(
    "what which how why where when who whom whose is are was were do does did "
    "the a an and or but in on at to for of with from by as not this that "
    "these those it its they them their has have had can could would should "
    "will may must shall be been being about also more than some other used using "
    "paper model method approach".split()
)


def extract_keyword_sentences(full_text: str, question: str, max_sentences: int = 8) -> list[str]:
    """Find sentences in full_text matching question keywords (BM25-like sentence retrieval)."""
    keywords = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", question) if w.lower() not in _KW_STOP]
    if not keywords:
        return []
    sentences = nltk.sent_tokenize(full_text)
    scored = []
    for sent in sentences:
        if len(sent) < 20 or len(sent) > 500:
            continue
        sent_lower = sent.lower()
        score = sum(1 for kw in keywords if kw in sent_lower)
        if score > 0:
            scored.append((score, sent))
    scored.sort(key=lambda x: -x[0])
    # Deduplicate near-identical sentences
    seen = set()
    result = []
    for _, sent in scored:
        key = sent[:60].lower()
        if key not in seen:
            seen.add(key)
            result.append(sent)
        if len(result) >= max_sentences:
            break
    return result


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

    def process_paper(self, entry: dict, paper_idx: int, total_papers: int) -> QAResult:
        title = entry["title"]
        full_text = entry["full_text"]
        question = entry["question"]

        self.logger.debug(f"\n{'='*70}")
        self.logger.debug(f"Paper #{paper_idx}/{total_papers}")
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

        # Stage 6b: Build expanded LLM context (decoupled from evidence output)
        reranked_pool = retrieval_debug.get("reranked_pool", retrieved)

        # Stage 6c: Keyword-matched sentences from full text (supplements retrieval)
        kw_sentences = extract_keyword_sentences(full_text, question, max_sentences=8)
        self.logger.debug(f"[LLM Context] reranked_pool={len(reranked_pool)}, kw_sentences={len(kw_sentences)}")

        # Stage 7: LLM Generation with self-consistency voting
        t0 = time.time()
        if self.enable_generation and self.generator is not None:
            answer, gen_debug = self.generator.generate(
                title, question, retrieved,
                extra_chunks=reranked_pool[len(retrieved):],
                extra_sentences=kw_sentences,
            )
            # Stage 7b: IDK / empty / terse Recovery
            # Treat these as degenerate and retry with permissive prompt:
            #   - IDK phrasing
            #   - Empty after cleaning
            #   - Very short fragment with no verb (e.g., "English.", "GPT", "KAR")
            def _needs_recovery(a: str) -> bool:
                if not a or not a.strip():
                    return True
                if Generator.is_idk(a):
                    return True
                stripped = a.strip().rstrip('.').strip()
                if len(stripped) < 12 and not re.search(r'\d', stripped):
                    # Too short to be a semantic answer unless it contains a number.
                    return True
                return False

            if _needs_recovery(answer):
                self.logger.debug(f"[Recovery] Detected weak answer ({answer!r}), retrying with fallback prompt")
                retry_answer, _ = self.generator.generate_fallback(
                    title, question, retrieved,
                    extra_chunks=reranked_pool[len(retrieved):],
                    extra_sentences=kw_sentences,
                )
                if retry_answer and not Generator.is_idk(retry_answer) and retry_answer.strip():
                    answer = retry_answer
                    self.logger.debug(f"[Recovery] Recovered: {answer[:100]}")
                else:
                    self.logger.debug(f"[Recovery] Fallback also failed, keeping original")
            dt_llm = time.time() - t0
        else:
            answer, gen_debug = "", {}
            dt_llm = 0.0
            self.logger.debug("[LLM] generation disabled")

        # Stage 8: Sentence-level evidence refinement
        # Decouple evidence output from chunk retrieval:
        #   - LLM gets full parent_text chunks for context (above)
        #   - Evidence output uses fine-grained sentence spans for ROUGE-L
        t0_refine = time.time()
        # K=1 by default: scoring formula is sum(ROUGE-L)/K, so a weak 2nd evidence kills score
        # Only K=2 for list-type questions where multiple golden evidences are expected
        refine_max_k = 2 if any(kw in question.lower() for kw in [
            "what are the", "list", "what methods", "what techniques",
            "what datasets", "what models", "what features",
            "what two", "what three", "what types", "what baselines",
        ]) else 1
        evidence = self.retriever.refine_evidence(
            question, reranked_pool, full_text,
            max_k=refine_max_k, logger=self.logger,
        )
        dt_refine = time.time() - t0_refine

        # Stage 9: Cleanup
        self.retriever.clear()

        total_time = time.time() - t_start

        self.logger.debug(f"[Result] answer ({len(answer)} chars): {answer}")
        self.logger.debug(f"[Result] evidence: {len(evidence)} refined spans")
        self.logger.debug(f"[Time] total={total_time:.1f}s "
                          f"(chunk={dt_chunk:.1f}s, index={dt_index:.1f}s, "
                          f"hyde={dt_hyde:.1f}s, retrieval={dt_retrieval:.1f}s, "
                          f"refine={dt_refine:.1f}s, llm={dt_llm:.1f}s)")

        return QAResult(title=title, answer=answer, evidence=evidence)

    def run(self, dataset: list) -> list:
        results = []
        t_all = time.time()

        pbar = tqdm(dataset, desc="Processing papers", unit="paper",
                    dynamic_ncols=True, file=sys.stdout)
        for paper_idx, entry in enumerate(pbar, start=1):
            short_title = entry["title"][:45] + "..." if len(entry["title"]) > 45 else entry["title"]
            pbar.set_postfix_str(short_title)
            result = self.process_paper(entry, paper_idx, len(dataset))
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

    if args.eval:
        if args.output:
            # --eval respects --output but forces it into outputs/ folder
            out_name = Path(args.output).name
            output_path = script_dir / "outputs" / out_name
        else:
            output_path = script_dir / "outputs" / "public.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output) if args.output else script_dir / f"{STUDENT_ID}.json"

    api_key = LLM_API_KEY
    if LLM_PROVIDER == "openrouter" and not api_key:
        tqdm.write("ERROR: Set OPENROUTER_API_KEY or LLM_API_KEY environment variable.")
        sys.exit(1)

    tqdm.write(f"{'#'*70}")
    tqdm.write(f"# HW2 RAG")
    tqdm.write(f"{'#'*70}")
    if args.eval and args.output:
        tqdm.write(f"  Note     : --eval writes to outputs/ folder → {output_path}")
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

    output = [{"title": r.title, "answer": r.answer, "evidence": r.evidence} for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tqdm.write(f"\nSaved {len(output)} results → {output_path}")

    if not args.eval:
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
