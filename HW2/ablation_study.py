#!/usr/bin/env python3
"""
Ablation Study: Find optimal chunking parameters for Evidence Score
Tunes: child_target, child_max, parent_window, parent_max

Usage:
  python ablation_study.py [--sample N] [--seed S] [--output results.csv]

Outputs:
  - CSV file with all combinations and their scores
  - JSON file with detailed results
  - Console summary of top configs
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import faiss
import nltk
from openai import OpenAI
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from dotenv import load_dotenv

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
# Ablation Configuration — EDIT HERE to change ranges
# ═══════════════════════════════════════════════════════════════════════════════
# 定義各個參數要測試的範圍 (組合數會相乘，請根據電腦效能調整)
CHILD_TARGETS = [150, 200, 250, 300]     # Ideal chunk size
CHILD_MAXS = [250, 400]             # Absolute max chunk size
PARENT_WINDOWS = [1, 2, 3]               # Chunks before + after
PARENT_MAXS = [600, 800, 1000]           # Max characters for LLM context window

EVAL_SAMPLE_SIZE = 20                    # Use 100 random papers for fast feedback
SEED = 42

# Fixed parameters (same as main pipeline)
DENSE_TOP_K = 40
BM25_TOP_K = 40
RRF_TOP_K = 20
RRF_K = 60
DEFAULT_FINAL_K = 5
MAX_EVIDENCE_FOR_LLM = 3

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "meta-llama/llama-3.2-3b-instruct"
LLM_BASE_URL = "https://openrouter.ai/api/v1"

LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 256

_STOP_WORDS = frozenset(
    "the and for are was were has have had does did can could would should "
    "this that these those with from into what which how why where when who "
    "whom whose not been about also more than their they them some other "
    "will may must shall being used using".split()
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes & Utilities
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Chunk:
    chunk_id: int
    text: str
    parent_text: str
    sent_start: int
    sent_end: int


@dataclass
class QAResult:
    title: str
    answer: str
    evidence: list


# ═══════════════════════════════════════════════════════════════════════════════
# DocumentProcessor (Dynamic chunk params)
# ═══════════════════════════════════════════════════════════════════════════════
class DocumentProcessor:
    def __init__(self, child_target: int, child_max: int, parent_window: int, parent_max: int):
        self.child_target = child_target
        self.child_max = child_max
        self.parent_window = parent_window
        self.parent_max = parent_max

    def split_sentences(self, text: str) -> list:
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
        sentences = self.split_sentences(full_text)
        if not sentences:
            return [Chunk(0, full_text[: self.child_max], full_text[: self.parent_max], 0, 0)]

        child_groups = []
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

        chunks = []
        for idx, (s, e, text) in enumerate(final_groups):
            pw_lo = max(0, idx - self.parent_window)
            pw_hi = min(len(final_groups), idx + self.parent_window + 1)
            parent = " ".join(g[2] for g in final_groups[pw_lo:pw_hi])
            if len(parent) > self.parent_max:
                parent = parent[: self.parent_max]
            chunks.append(Chunk(chunk_id=idx, text=text, parent_text=parent,
                                sent_start=s, sent_end=e))
        return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Retriever (same as main, silent mode)
# ═══════════════════════════════════════════════════════════════════════════════
class Retriever:
    def __init__(self, embed_model, reranker_model):
        self.embed_model = embed_model
        self.reranker = reranker_model
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
        if not ids:
            return []
        pairs = [(query, self.chunks[i].parent_text) for i in ids]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(ids, scores), key=lambda x: -x[1])
        return ranked[:top_k]

    def retrieve(self, question: str, variants: list, final_k: int) -> list:
        if not self.chunks:
            return []
        all_dense, all_bm25 = [], []
        for q in variants:
            all_dense.extend(self.dense_search(q, DENSE_TOP_K))
            all_bm25.extend(self.bm25_search(q, BM25_TOP_K))
        fused = self.rrf_fuse(all_dense, all_bm25)[:RRF_TOP_K]
        if not fused:
            return self.chunks[:final_k]
        reranked = self.rerank(question, fused, final_k)
        return [self.chunks[i] for i, _ in reranked]

    def clear(self):
        self.chunks, self.index, self.bm25 = [], None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Generator (silent mode, no LLM calls during ablation)
# ═══════════════════════════════════════════════════════════════════════════════
class Generator:
    # Generator 結構留存，但在此次評估腳本中暫不呼叫以節省時間與花費
    pass


def generate_query_variants(question: str) -> list:
    variants = [question]
    stripped = re.sub(
        r"^(what|which|how|why|where|when|who|does|do|did|is|are|was|were|can|could)"
        r"\s+(is|are|was|were|does|do|did|the|a|an|this|that)?\s*",
        "", question, flags=re.IGNORECASE
    ).strip().rstrip("?").strip()
    if stripped and stripped.lower() != question.lower() and len(stripped) > 10:
        variants.append(stripped)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", question)
    kw = [w for w in words if w.lower() not in _STOP_WORDS]
    if len(kw) >= 2:
        variants.append(" ".join(kw))
    return variants[:3]


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator
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


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
class AblationPipeline:
    def __init__(self, api_key: str, embed_model, reranker_model):
        self.api_key = api_key
        self.embed_model = embed_model
        self.reranker_model = reranker_model
        self.evaluator = Evaluator()

    @staticmethod
    def determine_k(question: str) -> int:
        q = question.lower()
        list_kw = [
            "list", "what are the", "which", "name the", "how many",
            "what methods", "what techniques", "what datasets",
            "what approaches", "what models", "what features",
        ]
        return 8 if any(k in q for k in list_kw) else DEFAULT_FINAL_K

    def process_paper(self, entry: dict, processor: DocumentProcessor, retriever: Retriever) -> float:
        full_text = entry["full_text"]
        question = entry["question"]

        chunks = processor.build_chunks(full_text)
        if not chunks:
            return 0.0

        retriever.build_index(chunks)

        variants = generate_query_variants(question)
        k = self.determine_k(question)
        retrieved = retriever.retrieve(question, variants, k)
        retriever.clear()

        golden_ev = entry.get("evidence", [])
        score = self.evaluator.evidence_score_single([c.text for c in retrieved], golden_ev)

        return score

    def run_config(
        self,
        dataset: list,
        child_target: int,
        child_max: int,
        parent_window: int,
        parent_max: int,
        show_paper_progress: bool = False,
        config_idx: int = 0,
        total_configs: int = 0,
        status_every: int = 5,
    ) -> dict:
        """Run evaluation for one config across all dataset papers."""
        scores = []
        processor = DocumentProcessor(child_target, child_max, parent_window, parent_max)
        retriever = Retriever(self.embed_model, self.reranker_model)
        n_papers = len(dataset)
        t0 = time.time()

        if show_paper_progress:
            desc_str = f"  ct={child_target} cm={child_max} pw={parent_window} pm={parent_max}"
            iterator = tqdm(dataset, desc=desc_str, unit="paper", leave=False, dynamic_ncols=True)
            for entry in iterator:
                score = self.process_paper(entry, processor, retriever)
                scores.append(score)
        else:
            if total_configs > 0 and config_idx > 0:
                tqdm.write(f"[Config {config_idx}/{total_configs}]")
                tqdm.write(f"  params: ct={child_target} cm={child_max} pw={parent_window} pm={parent_max}")
            for i, entry in enumerate(dataset, 1):
                score = self.process_paper(entry, processor, retriever)
                scores.append(score)
                if status_every > 0 and (i % status_every == 0 or i == n_papers):
                    elapsed = time.time() - t0
                    tqdm.write(
                        f"  papers: {i}/{n_papers} "
                        f"(elapsed {elapsed:.1f}s, avg {elapsed/max(i, 1):.2f}s/paper)"
                    )

        return {
            "child_target": child_target,
            "child_max": child_max,
            "parent_window": parent_window,
            "parent_max": parent_max,
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "n": len(scores),
            "scores": scores,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Tuning all 4 Chunking Parameters")
    parser.add_argument("--sample", type=int, default=EVAL_SAMPLE_SIZE,
                        help=f"Papers to sample (0=all, default={EVAL_SAMPLE_SIZE})")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--output", type=str, default="outputs/ablation_results.csv",
                        help="Output CSV file path")
    parser.add_argument(
        "--paper-progress",
        action="store_true",
        help="Show per-paper progress bar for each config (slower due to frequent terminal updates)",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=5,
        help="When --paper-progress is off, print status every N papers (0=disable, default=5)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # Load API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "ollama")
    if not api_key:
        tqdm.write("ERROR: Set OPENROUTER_API_KEY")
        sys.exit(1)

    # Load dataset
    dataset_path = script_dir / "public_dataset.json"
    if not dataset_path.exists():
        dataset_path = script_dir / "datasets" / "public_dataset.json"

    with open(dataset_path) as f:
        dataset = json.load(f)

    # Sample if requested
    if args.sample > 0 and args.sample < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.sample)
        tqdm.write(f"Sampled {len(dataset)} papers (seed={args.seed})")

    tqdm.write(f"Loaded {len(dataset)} papers from {dataset_path.name}\n")

    # Load models (once, reused across configs)
    tqdm.write("Loading models...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    reranker_model = CrossEncoder(RERANKER_MODEL)
    tqdm.write(f"  Embedding: {EMBED_MODEL}")
    tqdm.write(f"  Reranker:  {RERANKER_MODEL}\n")

    pipeline = AblationPipeline(api_key, embed_model, reranker_model)

    # Generate valid configurations
    valid_configs = []
    for ct, cm, pw, pm in itertools.product(CHILD_TARGETS, CHILD_MAXS, PARENT_WINDOWS, PARENT_MAXS):
        if cm <= ct:
            continue # 不合理的組合 (max小於或等於target) 予以略過
        valid_configs.append((ct, cm, pw, pm))

    total_configs = len(valid_configs)

    tqdm.write(f"Ablation Study Parameters:")
    tqdm.write(f"  CHILD_TARGETS : {CHILD_TARGETS}")
    tqdm.write(f"  CHILD_MAXS    : {CHILD_MAXS}")
    tqdm.write(f"  PARENT_WINDOWS: {PARENT_WINDOWS}")
    tqdm.write(f"  PARENT_MAXS   : {PARENT_MAXS}")
    tqdm.write(f"  Total valid configurations to run: {total_configs}\n")
    if not args.paper_progress:
        tqdm.write(f"Per-paper progress: OFF (faster), status every {args.status_every} papers\n")

    results = []
    best_mean = float("-inf")

    with tqdm(total=total_configs, desc="Overall Progress", unit="config") as pbar_overall:
        for idx, (ct, cm, pw, pm) in enumerate(valid_configs, 1):
            config_t0 = time.time()
            result = pipeline.run_config(
                dataset,
                ct,
                cm,
                pw,
                pm,
                show_paper_progress=args.paper_progress,
                config_idx=idx,
                total_configs=total_configs,
                status_every=args.status_every,
            )
            results.append(result)
            pbar_overall.update(1)
            if result["mean_score"] > best_mean:
                best_mean = result["mean_score"]
            pbar_overall.set_postfix_str(f"best={best_mean:.5f}")
            tqdm.write(
                f"[Config {idx}/{total_configs}] "
                f"mean_score={result['mean_score']:.5f} "
                f"(elapsed {time.time() - config_t0:.1f}s)"
            )

    # Sort by mean score descending
    results.sort(key=lambda r: r["mean_score"], reverse=True)

    # Save CSV
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "child_target", "child_max", "parent_window", "parent_max",
            "mean_score", "min_score", "max_score", "std_dev", "n"
        ])
        writer.writeheader()

        import statistics
        for rank, r in enumerate(results, 1):
            std = statistics.stdev(r["scores"]) if len(r["scores"]) > 1 else 0.0
            writer.writerow({
                "rank": rank,
                "child_target": r["child_target"],
                "child_max": r["child_max"],
                "parent_window": r["parent_window"],
                "parent_max": r["parent_max"],
                "mean_score": round(r["mean_score"], 5),
                "min_score": round(r["min_score"], 5),
                "max_score": round(r["max_score"], 5),
                "std_dev": round(std, 5),
                "n": r["n"],
            })

    # Save JSON with detailed scores
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump([
            {
                "rank": rank,
                "child_target": r["child_target"],
                "child_max": r["child_max"],
                "parent_window": r["parent_window"],
                "parent_max": r["parent_max"],
                "mean_score": r["mean_score"],
                "scores": [round(s, 5) for s in r["scores"]],
            }
            for rank, r in enumerate(results, 1)
        ], f, indent=2)

    # Print summary
    tqdm.write("\n" + "="*70)
    tqdm.write("TOP 5 CONFIGURATIONS")
    tqdm.write("="*70)
    for rank, r in enumerate(results[:5], 1):
        tqdm.write(
            f"{rank}. Target={r['child_target']:>3}  Max={r['child_max']:>3}  "
            f"Win={r['parent_window']}  P_Max={r['parent_max']:>4}  |  "
            f"Score={r['mean_score']:.5f}"
        )

    tqdm.write("="*70)
    tqdm.write(f"\nResults saved:")
    tqdm.write(f"  CSV:  {output_path}")
    tqdm.write(f"  JSON: {json_path}")

    # Comparison with baselines
    best = results[0]
    tqdm.write(f"\nBest configuration:")
    tqdm.write(f"  CHILD_TARGET_CHARS   = {best['child_target']}")
    tqdm.write(f"  CHILD_MAX_CHARS      = {best['child_max']}")
    tqdm.write(f"  PARENT_WINDOW_CHUNKS = {best['parent_window']}")
    tqdm.write(f"  PARENT_MAX_CHARS     = {best['parent_max']}")
    tqdm.write(f"  Evidence Score       = {best['mean_score']:.5f}")

    tqdm.write(f"\nComparison with baselines:")
    tqdm.write(f"  Weak baseline       : 0.2124")
    tqdm.write(f"  Strong baseline     : 0.26185")
    tqdm.write(f"  Best ablation       : {best['mean_score']:.5f}")
    if best["mean_score"] > 0.26185:
        tqdm.write(f"  ✓ Beats strong baseline by {(best['mean_score'] - 0.26185):.5f}")
    elif best["mean_score"] > 0.2124:
        tqdm.write(f"  ✓ Beats weak baseline by {(best['mean_score'] - 0.2124):.5f}")
    else:
        tqdm.write(f"  ✗ Below weak baseline, needs tuning")


if __name__ == "__main__":
    main()
