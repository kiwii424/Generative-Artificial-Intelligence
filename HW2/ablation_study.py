#!/usr/bin/env python3
"""
Comprehensive hyperparameter tuner for HW2 RAG.

This script does not modify `111511157.py`.
Instead, it dynamically loads the main pipeline and overrides score-related
hyperparameters from the outside, so the submission file stays untouched.

Features:
  - Edit parameter ranges directly in this file
  - Grid search or random search
  - Evidence-only mode (fast) or full-pipeline mode
  - Optional official public scoring via `score_public.py`

Examples:
  uv run python HW2/ablation_study.py
  uv run python HW2/ablation_study.py --strategy random --max-trials 20
  uv run python HW2/ablation_study.py --mode full --official-score --optimize official_combined_score
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import logging
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAIN_SCRIPT = SCRIPT_DIR / "111511157.py"
DEFAULT_DATASET = SCRIPT_DIR / "datasets" / "public_dataset.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "tuning"

# Edit ranges here directly.
# Each value must be a list. Grid search tests all combinations; random search
# samples from these lists.
SEARCH_SPACE: dict[str, list[Any]] = {
    # Models
    "embed_model": ["BAAI/bge-large-en-v1.5"],
    "reranker_model": ["BAAI/bge-reranker-v2-m3"],
    "llm_model": ["meta-llama/llama-3.2-3b-instruct"],  # fixed by course rule

    # Chunking
    "child_target_chars": [200],
    "child_max_chars": [400],
    "parent_window_chunks": [2],
    "parent_max_chars": [800],
    "chunk_sentence_overlap": [1],
    "sentence_merge_min_chars": [60],
    "force_split_window": [80],

    # Retrieval / fusion / reranking
    "dense_top_k": [40],
    "bm25_top_k": [40],
    "rrf_top_k": [20],
    "rrf_k": [60],
    "rerank_pool": [15],
    "reranker_score_threshold": [0.0],
    "reranker_gap_threshold": [2.0],
    "reranker_relative_threshold": [0.3],
    "section_boost_score": [1.5],
    "enable_section_routing": [True],
    "reranker_max_length": [1024],
    "embed_query_prefix": ["Represent this sentence for searching relevant passages: "],

    # Query expansion / HyDE
    "enable_stripped_query": [True],
    "enable_keyword_query": [True],
    "max_query_variants": [3],
    "keyword_variant_min_terms": [2],
    "stripped_query_min_chars": [10],
    "enable_hyde": [True],
    "hyde_temperature": [0.3],
    "hyde_max_tokens": [150],

    # Evidence selection
    "default_final_k": [2],
    "list_question_final_k": [3],
    "max_evidence_for_llm": [5],

    # Generation
    "llm_temperature": [0.1],
    "llm_max_tokens": [256],
}

OPTIMIZE_CHOICES = [
    "mean_evidence_score",
    "official_evidence_score",
    "official_correctness",
    "official_combined_score",
]

LLM_ONLY_IN_EVIDENCE_MODE = {
    "max_evidence_for_llm",
    "llm_temperature",
    "llm_max_tokens",
}

LIST_QUESTION_KEYWORDS = [
    "list", "what are the", "name the", "how many",
    "what methods", "what techniques", "what datasets",
    "what approaches", "what models", "what features",
    "what types", "what kinds", "what categories",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune HW2 hyperparameters by editing SEARCH_SPACE in this file."
    )
    parser.add_argument("--main-script", type=str, default=str(DEFAULT_MAIN_SCRIPT),
                        help="Path to the main submission script (default: HW2/111511157.py)")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET),
                        help="Dataset JSON with ground-truth evidence/answer (default: public dataset)")
    parser.add_argument("--mode", choices=["evidence", "full"], default="evidence",
                        help="evidence=fast retrieval-only scoring, full=run the full pipeline")
    parser.add_argument("--strategy", choices=["grid", "random"], default="grid",
                        help="Search strategy")
    parser.add_argument("--max-trials", type=int, default=20,
                        help="Maximum trials for random search (default: 20)")
    parser.add_argument("--sample", type=int, default=20,
                        help="Randomly sample N papers from dataset (0=all, default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--optimize", choices=OPTIMIZE_CHOICES, default="mean_evidence_score",
                        help="Metric to optimize")
    parser.add_argument("--official-score", action="store_true",
                        help="In full mode, additionally run score_public.py for official-style evidence/correctness")
    parser.add_argument("--judge-host", type=str, default="localhost",
                        help="Host for score_public.py judge server")
    parser.add_argument("--judge-port", type=int, default=8091,
                        help="Port for score_public.py judge server")
    parser.add_argument("--judge-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Judge model name for score_public.py")
    parser.add_argument("--judge-times", type=int, default=5,
                        help="Judge runs per paper for score_public.py")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for CSV/JSON summaries")
    parser.add_argument("--keep-trial-files", action="store_true",
                        help="Keep per-trial prediction and score JSON files")
    parser.add_argument("--verbose", action="store_true", help="Print more logs to the console")
    return parser.parse_args()


def resolve_dataset_path(dataset_arg: str) -> Path:
    path = Path(dataset_arg)
    if path.exists():
        return path
    candidate = SCRIPT_DIR / dataset_arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Dataset not found: {dataset_arg}")


def load_dataset(path: Path, sample: int, seed: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        dataset = json.load(f)

    if sample > 0 and sample < len(dataset):
        rng = random.Random(seed)
        dataset = rng.sample(dataset, sample)
    return dataset


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def setup_logger(output_dir: Path, verbose: bool) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"tuner_{timestamp}.log"

    logger = logging.getLogger(f"rag_tuner_{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    return logger, log_path


def grid_size(space: dict[str, list[Any]]) -> int:
    return math.prod(len(values) for values in space.values())


def validate_config(config: dict[str, Any]) -> tuple[bool, str]:
    positive_int_fields = [
        "child_target_chars", "child_max_chars", "parent_window_chunks", "parent_max_chars",
        "chunk_sentence_overlap", "sentence_merge_min_chars", "force_split_window",
        "dense_top_k", "bm25_top_k", "rrf_top_k", "rrf_k", "rerank_pool", "reranker_max_length",
        "max_query_variants", "keyword_variant_min_terms", "stripped_query_min_chars",
        "hyde_max_tokens", "default_final_k", "list_question_final_k",
        "max_evidence_for_llm", "llm_max_tokens",
    ]
    for field in positive_int_fields:
        if int(config[field]) <= 0:
            return False, f"{field} must be > 0"

    if int(config["child_max_chars"]) <= int(config["child_target_chars"]):
        return False, "child_max_chars must be > child_target_chars"
    if int(config["default_final_k"]) > 40 or int(config["list_question_final_k"]) > 40:
        return False, "submission evidence count cannot exceed 40"
    if int(config["max_evidence_for_llm"]) > 40:
        return False, "max_evidence_for_llm cannot exceed 40"
    return True, ""


def iter_grid_configs(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(space.keys())
    configs = []
    for combo in itertools.product(*(space[key] for key in keys)):
        config = dict(zip(keys, combo, strict=True))
        ok, _ = validate_config(config)
        if ok:
            configs.append(config)
    return configs


def iter_random_configs(space: dict[str, list[Any]], max_trials: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    configs = []
    seen = set()
    max_attempts = max(max_trials * 50, 100)

    for _ in range(max_attempts):
        if len(configs) >= max_trials:
            break
        config = {key: rng.choice(values) for key, values in space.items()}
        ok, _ = validate_config(config)
        if not ok:
            continue
        sig = json.dumps(config, sort_keys=True, ensure_ascii=False)
        if sig in seen:
            continue
        seen.add(sig)
        configs.append(config)

    return configs


def build_configs(space: dict[str, list[Any]], strategy: str, max_trials: int, seed: int) -> list[dict[str, Any]]:
    if strategy == "grid":
        return iter_grid_configs(space)
    return iter_random_configs(space, max_trials=max_trials, seed=seed)


def warn_ignored_params(space: dict[str, list[Any]], mode: str) -> None:
    if mode != "evidence":
        return
    ignored = [key for key in sorted(LLM_ONLY_IN_EVIDENCE_MODE) if len(space[key]) > 1]
    if ignored:
        tqdm.write(
            "Warning: these params are varied but do not affect evidence-only mode: "
            + ", ".join(ignored)
        )


def load_main_module(main_script: Path):
    spec = importlib.util.spec_from_file_location("hw2_main_module", main_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {main_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ModelCache:
    def __init__(self):
        self.embed_models: dict[str, Any] = {}
        self.rerankers: dict[tuple[str, int], Any] = {}

    def get_embed(self, module, model_name: str):
        if model_name not in self.embed_models:
            tqdm.write(f"[Init] Loading embedding model: {model_name}")
            self.embed_models[model_name] = module.SentenceTransformer(model_name)
        return self.embed_models[model_name]

    def get_reranker(self, module, model_name: str, max_length: int):
        key = (model_name, max_length)
        if key not in self.rerankers:
            tqdm.write(f"[Init] Loading reranker: {model_name} (max_length={max_length})")
            self.rerankers[key] = module.CrossEncoder(model_name, max_length=max_length)
        return self.rerankers[key]


def capture_originals(module) -> dict[str, Any]:
    return {
        "DocumentProcessor": module.DocumentProcessor,
        "Retriever": module.Retriever,
        "Generator": module.Generator,
        "RAGPipeline": module.RAGPipeline,
        "generate_query_variants": module.generate_query_variants,
        "classify_question_sections": module.classify_question_sections,
    }


def apply_global_overrides(module, config: dict[str, Any]) -> None:
    module.EMBED_MODEL = config["embed_model"]
    module.RERANKER_MODEL = config["reranker_model"]
    module.LLM_MODEL = config["llm_model"]

    module.CHILD_TARGET_CHARS = int(config["child_target_chars"])
    module.CHILD_MAX_CHARS = int(config["child_max_chars"])
    module.PARENT_WINDOW_CHUNKS = int(config["parent_window_chunks"])
    module.PARENT_MAX_CHARS = int(config["parent_max_chars"])
    module.CHUNK_SENTENCE_OVERLAP = int(config["chunk_sentence_overlap"])

    module.DENSE_TOP_K = int(config["dense_top_k"])
    module.BM25_TOP_K = int(config["bm25_top_k"])
    module.RRF_TOP_K = int(config["rrf_top_k"])
    module.RRF_K = int(config["rrf_k"])
    module.DEFAULT_FINAL_K = int(config["default_final_k"])
    module.RERANK_POOL = int(config["rerank_pool"])

    module.RERANKER_SCORE_THRESHOLD = float(config["reranker_score_threshold"])
    module.RERANKER_GAP_THRESHOLD = float(config["reranker_gap_threshold"])
    module.SECTION_BOOST_SCORE = float(config["section_boost_score"])

    module.MAX_EVIDENCE_FOR_LLM = int(config["max_evidence_for_llm"])
    module.LLM_TEMPERATURE = float(config["llm_temperature"])
    module.LLM_MAX_TOKENS = int(config["llm_max_tokens"])


def install_overrides(module, originals: dict[str, Any], config: dict[str, Any], model_cache: ModelCache) -> None:
    apply_global_overrides(module, config)

    BaseDocumentProcessor = originals["DocumentProcessor"]
    BaseRetriever = originals["Retriever"]
    BaseGenerator = originals["Generator"]
    BaseRAGPipeline = originals["RAGPipeline"]
    base_classify_sections = originals["classify_question_sections"]

    sentence_merge_min_chars = int(config["sentence_merge_min_chars"])
    force_split_window = int(config["force_split_window"])
    query_prefix = str(config["embed_query_prefix"])
    relative_threshold = float(config["reranker_relative_threshold"])
    reranker_max_length = int(config["reranker_max_length"])
    enable_section_routing = bool(config["enable_section_routing"])
    enable_hyde = bool(config["enable_hyde"])
    hyde_temperature = float(config["hyde_temperature"])
    hyde_max_tokens = int(config["hyde_max_tokens"])
    enable_stripped_query = bool(config["enable_stripped_query"])
    enable_keyword_query = bool(config["enable_keyword_query"])
    max_query_variants = int(config["max_query_variants"])
    keyword_variant_min_terms = int(config["keyword_variant_min_terms"])
    stripped_query_min_chars = int(config["stripped_query_min_chars"])
    list_question_final_k = int(config["list_question_final_k"])

    class ConfigurableDocumentProcessor(BaseDocumentProcessor):
        def split_sentences(self, text: str) -> list[str]:
            raw = module.nltk.sent_tokenize(text)
            merged = []
            buf = ""
            for sentence in raw:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if buf:
                    buf = buf + " " + sentence
                    if len(buf) >= sentence_merge_min_chars:
                        merged.append(buf)
                        buf = ""
                elif len(sentence) < sentence_merge_min_chars:
                    buf = sentence
                else:
                    merged.append(sentence)
            if buf:
                if merged:
                    merged[-1] = merged[-1] + " " + buf
                else:
                    merged.append(buf)
            return merged

        def _group_sentences_to_chunks(self, sentences: list[str]) -> list[tuple[int, int, str]]:
            if not sentences:
                return []

            child_groups = []
            current_sents = []
            start_idx = 0

            for i, sent in enumerate(sentences):
                current_text = " ".join(current_sents + [sent])
                if current_sents and len(current_text) > self.child_target:
                    child_groups.append((start_idx, i - 1, " ".join(current_sents)))
                    overlap_start = max(0, len(current_sents) - self.sentence_overlap)
                    overlap_sents = current_sents[overlap_start:]
                    current_sents = overlap_sents + [sent]
                    start_idx = i - len(overlap_sents)
                else:
                    current_sents.append(sent)

            if current_sents:
                child_groups.append((start_idx, len(sentences) - 1, " ".join(current_sents)))

            final_groups = []
            for s, e, text in child_groups:
                if len(text) <= self.child_max:
                    final_groups.append((s, e, text))
                    continue

                mid = len(text) // 2
                sp = text.rfind(". ", max(0, mid - force_split_window), mid + force_split_window)
                if sp == -1:
                    sp = text.rfind(", ", max(0, mid - force_split_window), mid + force_split_window)
                if sp == -1:
                    sp = mid
                final_groups.append((s, e, text[: sp + 1].strip()))
                final_groups.append((s, e, text[sp + 1 :].strip()))
            return final_groups

    class ConfigurableRetriever(BaseRetriever):
        def __init__(self, logger: logging.Logger):
            self.logger = logger
            self.embed_model = model_cache.get_embed(module, module.EMBED_MODEL)
            self.reranker = model_cache.get_reranker(module, module.RERANKER_MODEL, reranker_max_length)
            self.chunks = []
            self.index = None
            self.bm25 = None

        def _embed_query(self, q: str):
            prefixed = query_prefix + q
            return self.embed_model.encode([prefixed], normalize_embeddings=True).astype("float32")

        def select_dynamic_k(self, reranked: list[tuple[int, float]], max_k: int) -> list[tuple[int, float]]:
            if not reranked or len(reranked) <= 1:
                return reranked[:1] if reranked else []

            selected = [reranked[0]]
            top_score = reranked[0][1]

            for i in range(1, min(len(reranked), max_k)):
                _, score = reranked[i]
                prev_score = reranked[i - 1][1]

                if score < module.RERANKER_SCORE_THRESHOLD:
                    break
                if prev_score - score > module.RERANKER_GAP_THRESHOLD:
                    break
                if top_score > 0 and relative_threshold > 0 and score < top_score * relative_threshold:
                    break

                selected.append(reranked[i])

            return selected

    class ConfigurableGenerator(BaseGenerator):
        def generate_hyde(self, title: str, question: str) -> str:
            if not enable_hyde:
                self.logger.debug("[HyDE] disabled")
                return ""

            messages = [
                {"role": "system", "content":
                 "You are a scientific paper expert. Given a paper title and question, "
                 "write a brief passage (2-3 sentences) that might appear in the paper "
                 "as an answer. Write as if quoting the paper. Be specific and technical."},
                {"role": "user", "content": f"Paper: {title}\nQuestion: {question}\nPassage:"},
            ]
            try:
                resp = self.client.chat.completions.create(
                    model=module.LLM_MODEL,
                    messages=messages,
                    temperature=hyde_temperature,
                    max_tokens=hyde_max_tokens,
                )
                result = resp.choices[0].message.content.strip()
                self.logger.debug(f"[HyDE] Generated: {result[:200]}")
                return result
            except Exception as exc:
                self.logger.warning(f"[HyDE] Failed: {exc}")
                return ""

    def generate_query_variants(question: str) -> list[str]:
        variants = []
        seen = set()

        def add_variant(text: str) -> None:
            text = text.strip()
            if not text:
                return
            lowered = text.lower()
            if lowered in seen:
                return
            seen.add(lowered)
            variants.append(text)

        add_variant(question)

        if enable_stripped_query:
            stripped = re.sub(
                r"^(what|which|how|why|where|when|who|does|do|did|is|are|was|were|can|could)"
                r"\s+(is|are|was|were|does|do|did|the|a|an|this|that)?\s*",
                "",
                question,
                flags=re.IGNORECASE,
            ).strip().rstrip("?").strip()
            if len(stripped) >= stripped_query_min_chars:
                add_variant(stripped)

        if enable_keyword_query:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", question)
            keywords = [word for word in words if word.lower() not in module._STOP_WORDS]
            if len(keywords) >= keyword_variant_min_terms:
                add_variant(" ".join(keywords))

        return variants[:max_query_variants]

    def classify_question_sections(question: str) -> set[str]:
        if not enable_section_routing:
            return set()
        return base_classify_sections(question)

    def determine_k(question: str) -> int:
        q = question.lower()
        return list_question_final_k if any(keyword in q for keyword in LIST_QUESTION_KEYWORDS) else module.DEFAULT_FINAL_K

    module.DocumentProcessor = ConfigurableDocumentProcessor
    module.Retriever = ConfigurableRetriever
    module.Generator = ConfigurableGenerator
    module.generate_query_variants = generate_query_variants
    module.classify_question_sections = classify_question_sections
    module.RAGPipeline = BaseRAGPipeline
    module.RAGPipeline.determine_k = staticmethod(determine_k)


def ensure_dataset_has_labels(dataset: list[dict[str, Any]], optimize: str, official_score: bool) -> None:
    need_answers = official_score or optimize in {"official_correctness", "official_combined_score"}
    missing_evidence = any("evidence" not in item for item in dataset)
    missing_answer = any("answer" not in item for item in dataset)

    if missing_evidence:
        raise ValueError("Dataset must contain `evidence` for tuning/score evaluation.")
    if need_answers and missing_answer:
        raise ValueError("Dataset must contain `answer` when official correctness scoring is requested.")


def maybe_require_api_key(module, configs: list[dict[str, Any]], mode: str) -> str:
    llm_needed = mode == "full" or any(bool(cfg["enable_hyde"]) for cfg in configs)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    base_url = str(getattr(module, "LLM_BASE_URL", ""))

    if llm_needed and "openrouter" in base_url.lower() and not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is required because the tuned pipeline will call OpenRouter.")

    return api_key or "ollama"


def score_evidence_only(
    module,
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    logger: logging.Logger,
    api_key: str,
) -> dict[str, Any]:
    processor = module.DocumentProcessor(logger)
    retriever = module.Retriever(logger)
    generator = module.Generator(api_key=api_key, logger=logger) if bool(config["enable_hyde"]) else None
    evaluator = module.Evaluator()

    per_paper_scores = []
    predictions = []

    iterator = tqdm(dataset, desc="  Papers", unit="paper", leave=False, dynamic_ncols=True)
    for entry in iterator:
        chunks = processor.build_chunks(entry["full_text"])
        if not chunks:
            per_paper_scores.append(0.0)
            predictions.append({"title": entry["title"], "answer": "", "evidence": []})
            continue

        retriever.build_index(chunks)
        variants = module.generate_query_variants(entry["question"])
        target_sections = module.classify_question_sections(entry["question"])
        hyde_text = ""
        if generator is not None:
            hyde_text = generator.generate_hyde(entry["title"], entry["question"])

        final_k = module.RAGPipeline.determine_k(entry["question"])
        retrieved, _ = retriever.retrieve(
            entry["question"],
            variants,
            final_k,
            hyde_text=hyde_text,
            target_sections=target_sections,
        )
        retriever.clear()

        evidence = [chunk.text for chunk in retrieved]
        score = evaluator.evidence_score_single(evidence, entry.get("evidence", []))
        per_paper_scores.append(score)
        predictions.append({"title": entry["title"], "answer": "", "evidence": evidence})

    mean_score = statistics.mean(per_paper_scores) if per_paper_scores else 0.0
    return {
        "mean_evidence_score": mean_score,
        "min_evidence_score": min(per_paper_scores) if per_paper_scores else 0.0,
        "max_evidence_score": max(per_paper_scores) if per_paper_scores else 0.0,
        "std_evidence_score": statistics.pstdev(per_paper_scores) if len(per_paper_scores) > 1 else 0.0,
        "n": len(per_paper_scores),
        "per_paper_scores": per_paper_scores,
        "predictions": predictions,
    }


def score_full_pipeline(module, dataset: list[dict[str, Any]], logger: logging.Logger, api_key: str) -> dict[str, Any]:
    pipeline = module.RAGPipeline(api_key=api_key, logger=logger)
    evaluator = module.Evaluator()

    per_paper_scores = []
    predictions = []

    iterator = tqdm(dataset, desc="  Papers", unit="paper", leave=False, dynamic_ncols=True)
    for entry in iterator:
        result = pipeline.process_paper(entry)
        evidence_score = evaluator.evidence_score_single(result.evidence, entry.get("evidence", []))
        per_paper_scores.append(evidence_score)
        predictions.append({
            "title": result.title,
            "answer": result.answer,
            "evidence": result.evidence,
        })

    mean_score = statistics.mean(per_paper_scores) if per_paper_scores else 0.0
    return {
        "mean_evidence_score": mean_score,
        "min_evidence_score": min(per_paper_scores) if per_paper_scores else 0.0,
        "max_evidence_score": max(per_paper_scores) if per_paper_scores else 0.0,
        "std_evidence_score": statistics.pstdev(per_paper_scores) if len(per_paper_scores) > 1 else 0.0,
        "n": len(per_paper_scores),
        "per_paper_scores": per_paper_scores,
        "predictions": predictions,
    }


def run_official_score(
    prediction_path: Path,
    dataset_path: Path,
    judge_host: str,
    judge_port: int,
    judge_model: str,
    judge_times: int,
) -> dict[str, float]:
    score_script = SCRIPT_DIR / "score_public.py"
    cmd = [
        sys.executable,
        str(score_script),
        str(prediction_path),
        "--host", judge_host,
        "--port", str(judge_port),
        "--model", judge_model,
        "--dataset", str(dataset_path),
        "--times", str(judge_times),
    ]
    proc = subprocess.run(cmd, cwd=SCRIPT_DIR, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "score_public.py failed.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    score_json = prediction_path.with_name(prediction_path.stem + "_score.json")
    if not score_json.exists():
        raise FileNotFoundError(f"Official score JSON not found: {score_json}")

    with score_json.open(encoding="utf-8") as f:
        summary = json.load(f)["summary"]

    return {
        "official_evidence_score": float(summary["evidence_score"]),
        "official_correctness": float(summary["correctness"]),
        "official_combined_score": (float(summary["evidence_score"]) + float(summary["correctness"])) / 2.0,
    }


def objective_value(metrics: dict[str, Any], optimize: str) -> float:
    value = metrics.get(optimize)
    if value is None:
        raise KeyError(f"Metric {optimize} is not available for this run.")
    return float(value)


def save_trial_artifacts(
    output_dir: Path,
    trial_idx: int,
    predictions: list[dict[str, Any]],
    official_metrics: dict[str, float] | None,
    keep_trial_files: bool,
) -> None:
    if not keep_trial_files:
        return

    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    pred_path = trials_dir / f"trial_{trial_idx:03d}.json"
    dump_json(predictions, pred_path)

    if official_metrics is not None:
        dump_json(official_metrics, trials_dir / f"trial_{trial_idx:03d}_official_score.json")


def format_config(config: dict[str, Any]) -> str:
    return (
        f"ct={config['child_target_chars']} cm={config['child_max_chars']} "
        f"pw={config['parent_window_chunks']} pm={config['parent_max_chars']} ov={config['chunk_sentence_overlap']} "
        f"d/b/r={config['dense_top_k']}/{config['bm25_top_k']}/{config['rrf_top_k']} "
        f"pool={config['rerank_pool']} fk={config['default_final_k']} lk={config['list_question_final_k']} "
        f"hyde={'on' if config['enable_hyde'] else 'off'} sec={'on' if config['enable_section_routing'] else 'off'}"
    )


def save_results(output_dir: Path, optimize: str, results: list[dict[str, Any]], best_result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"tuning_results_{timestamp}.csv"
    json_path = output_dir / f"tuning_results_{timestamp}.json"
    best_path = output_dir / "best_config.json"
    best_space_path = output_dir / "best_search_space.json"

    metric_columns = [
        "objective_value",
        "mean_evidence_score",
        "min_evidence_score",
        "max_evidence_score",
        "std_evidence_score",
        "official_evidence_score",
        "official_correctness",
        "official_combined_score",
        "elapsed_seconds",
        "n",
    ]
    config_columns = list(SEARCH_SPACE.keys())
    fieldnames = ["rank", "trial"] + metric_columns + config_columns

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(results, 1):
            row = {key: item.get(key) for key in fieldnames}
            row["rank"] = rank
            writer.writerow(row)

    dump_json(
        {
            "optimize": optimize,
            "results": results,
            "best": best_result,
        },
        json_path,
    )
    dump_json(best_result, best_path)
    dump_json({key: [best_result[key]] for key in config_columns}, best_space_path)

    tqdm.write(f"\nSaved results:")
    tqdm.write(f"  CSV : {csv_path}")
    tqdm.write(f"  JSON: {json_path}")
    tqdm.write(f"  Best: {best_path}")
    tqdm.write(f"  Re-run best config with: {best_space_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger, log_path = setup_logger(output_dir, verbose=args.verbose)

    search_space = {key: list(values) for key, values in SEARCH_SPACE.items()}

    if args.official_score and args.mode != "full":
        raise ValueError("--official-score requires --mode full")
    if args.optimize != "mean_evidence_score" and not args.official_score:
        raise ValueError(f"--optimize {args.optimize} requires --official-score")

    warn_ignored_params(search_space, args.mode)

    dataset_path = resolve_dataset_path(args.dataset)
    dataset = load_dataset(dataset_path, sample=args.sample, seed=args.seed)
    ensure_dataset_has_labels(dataset, optimize=args.optimize, official_score=args.official_score)

    main_script = Path(args.main_script)
    if not main_script.exists():
        raise FileNotFoundError(f"Main script not found: {main_script}")

    total_grid = grid_size(search_space)
    configs = build_configs(search_space, strategy=args.strategy, max_trials=args.max_trials, seed=args.seed)
    if not configs:
        raise ValueError("No valid configurations generated from the search space.")

    tqdm.write(f"{'#' * 70}")
    tqdm.write("Comprehensive HW2 Hyperparameter Tuner")
    tqdm.write(f"{'#' * 70}")
    tqdm.write(f"  Main script : {main_script}")
    tqdm.write(f"  Dataset     : {dataset_path}")
    tqdm.write(f"  Mode        : {args.mode}")
    tqdm.write(f"  Optimize    : {args.optimize}")
    tqdm.write(f"  Strategy    : {args.strategy}")
    tqdm.write(f"  Sample      : {len(dataset)} papers")
    tqdm.write(f"  Grid size   : {total_grid}")
    tqdm.write(f"  Trials      : {len(configs)}")
    tqdm.write(f"  Output dir  : {output_dir}")
    tqdm.write(f"  Log file    : {log_path}\n")

    module = load_main_module(main_script)
    originals = capture_originals(module)
    model_cache = ModelCache()
    api_key = maybe_require_api_key(module, configs=configs, mode=args.mode)

    results = []
    best_result = None
    best_score = float("-inf")

    with tqdm(total=len(configs), desc="Overall Progress", unit="trial") as progress:
        for trial_idx, config in enumerate(configs, 1):
            install_overrides(module, originals, config, model_cache)
            logger.debug("Trial %s config: %s", trial_idx, json.dumps(config, ensure_ascii=False, sort_keys=True))

            tqdm.write(f"[Trial {trial_idx}/{len(configs)}] {format_config(config)}")
            start_time = time.time()

            if args.mode == "evidence":
                metrics = score_evidence_only(module, config=config, dataset=dataset, logger=logger, api_key=api_key)
            else:
                metrics = score_full_pipeline(module, dataset=dataset, logger=logger, api_key=api_key)

            official_metrics = None
            if args.official_score:
                if args.keep_trial_files:
                    prediction_path = output_dir / "trials" / f"trial_{trial_idx:03d}.json"
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    dump_json(metrics["predictions"], prediction_path)
                    official_metrics = run_official_score(
                        prediction_path=prediction_path,
                        dataset_path=dataset_path,
                        judge_host=args.judge_host,
                        judge_port=args.judge_port,
                        judge_model=args.judge_model,
                        judge_times=args.judge_times,
                    )
                else:
                    with tempfile.TemporaryDirectory(prefix="hw2_tuner_") as tmp_dir:
                        prediction_path = Path(tmp_dir) / f"trial_{trial_idx:03d}.json"
                        dump_json(metrics["predictions"], prediction_path)
                        official_metrics = run_official_score(
                            prediction_path=prediction_path,
                            dataset_path=dataset_path,
                            judge_host=args.judge_host,
                            judge_port=args.judge_port,
                            judge_model=args.judge_model,
                            judge_times=args.judge_times,
                        )

            elapsed = time.time() - start_time
            record = {
                "trial": trial_idx,
                "elapsed_seconds": round(elapsed, 3),
                **config,
                **{k: v for k, v in metrics.items() if k != "predictions" and k != "per_paper_scores"},
            }
            if official_metrics is not None:
                record.update(official_metrics)
            else:
                record.setdefault("official_evidence_score", None)
                record.setdefault("official_correctness", None)
                record.setdefault("official_combined_score", None)

            record["objective_value"] = round(objective_value(record, args.optimize), 6)
            results.append(record)

            save_trial_artifacts(
                output_dir=output_dir,
                trial_idx=trial_idx,
                predictions=metrics["predictions"],
                official_metrics=official_metrics,
                keep_trial_files=args.keep_trial_files,
            )

            if record["objective_value"] > best_score:
                best_score = record["objective_value"]
                best_result = record

            progress.update(1)
            progress.set_postfix_str(f"best={best_score:.5f}")

            summary = f"mean_evidence={record['mean_evidence_score']:.5f}"
            if record["official_combined_score"] is not None:
                summary += (
                    f", official_evidence={record['official_evidence_score']:.5f},"
                    f" official_correctness={record['official_correctness']:.5f},"
                    f" combined={record['official_combined_score']:.5f}"
                )
            tqdm.write(f"  -> {summary} (elapsed {elapsed:.1f}s)")

    if best_result is None:
        raise RuntimeError("No result was produced.")

    results.sort(key=lambda item: item["objective_value"], reverse=True)
    save_results(output_dir=output_dir, optimize=args.optimize, results=results, best_result=best_result)

    tqdm.write("\nTop 5 configurations:")
    for rank, item in enumerate(results[:5], 1):
        tqdm.write(
            f"{rank}. score={item['objective_value']:.5f}  "
            f"evidence={item['mean_evidence_score']:.5f}  "
            f"{format_config(item)}"
        )

    tqdm.write("\nBest configuration:")
    tqdm.write(json.dumps(best_result, indent=2, ensure_ascii=False))
    tqdm.write(f"\nTuner log saved -> {log_path}")


if __name__ == "__main__":
    main()
