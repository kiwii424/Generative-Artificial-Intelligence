#!/usr/bin/env python3
"""
Ablation Study for HW2 RAG Pipeline

Systematically tests different configurations to maximize evidence score (ROUGE-L).

Default LLM provider: openrouter
For local Ollama, prefix commands with LLM_PROVIDER=ollama

Usage:
  uv run python ablation_study.py                                               # Run all fast experiments
  uv run python ablation_study.py --experiment k                                # Single experiment
  
  uv run python ablation_study.py --experiment chunk_context_joint --sample 25 --max-configs-per-run 5   # Joint search: chunk + parent context
  uv run python ablation_study.py --experiment chunk_context_top10 --sample 25                           # Re-run top-10 chunk+context configs
  uv run python ablation_study.py --experiment retrieval_joint --sample 25 --max-configs-per-run 20       # Joint search: retrieval pool
  uv run python ablation_study.py --experiment selection_joint --sample 25 --max-configs-per-run 5       # Joint search: rerank/final evidence
  uv run python ablation_study.py --experiment finalists_joint                   # Final 12 configs on full public set
  
  uv run python ablation_study.py --experiment grid                             # Grid search
  uv run python ablation_study.py --experiment grid --max-configs-per-run 5     # Run 5 configs, save progress, then stop
  uv run python ablation_study.py --sample 10                                   # Fewer papers (faster)
  uv run python ablation_study.py --sample 0                                    # All 100 papers


  
Used Usage:
  uv run python ablation_study.py --experiment model_combos --sample 1          # All embed+reranker combos
  uv run python ablation_study.py --experiment model_combos_top10 --sample 1    # Top-10 selected model pairs
  

  
Experiments:
  k            - DEFAULT_FINAL_K = 1, 2, 3
  chunk_target - CHILD_TARGET_CHARS = 100, 150, 200, 250, 300
  chunk_max    - CHILD_MAX_CHARS = 300, 400, 500, 600
  overlap      - CHUNK_SENTENCE_OVERLAP = 0, 1, 2
  parent       - PARENT_WINDOW_CHUNKS = 1, 2, 3, 4
  threshold    - Reranker score + gap thresholds
  section      - SECTION_BOOST_SCORE = 0, 0.5, 1.0, 1.5, 2.0, 3.0
  rerank_pool  - RERANK_POOL = 10, 15, 20, 25, 30
  rrf_top_k    - RRF_TOP_K = 15, 20, 25, 30
  chunk_context_joint - CHILD_* × overlap × parent context
  chunk_context_top10 - Re-run top-10 configs from chunk_context_joint
  retrieval_joint - DENSE/BM25/RRF joint search
  selection_joint - rerank pool × final K × thresholds × section boost
  finalists_joint - 12 final configs from stage shortlists, run on full public set
  model_combos - All embedding + reranker model pairs
  model_combos_top10 - 10 selected embedding + reranker pairs
  grid         - Grid search: K × chunk_target × rerank_pool
  all          - Run all single-param experiments (skip slow model tests)
"""

import argparse
import itertools
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import importlib.util


ABLATION_BASELINE_OVERRIDES = {
    "CHILD_MAX_CHARS": 400,
    "CHILD_TARGET_CHARS": 200,
    "CHUNK_SENTENCE_OVERLAP": 1,
    "DEFAULT_FINAL_K": 1,
    "EMBED_MODEL": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "EMBED_PASSAGE_PREFIX": "Represent this sentence for retrieval: ",
    "EMBED_QUERY_PREFIX": "Represent this sentence for searching relevant passages: ",
    "RERANKER_GAP_THRESHOLD": 2.0,
    "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
    "RERANKER_SCORE_THRESHOLD": 0.0,
    "RERANK_POOL": 15,
    "RRF_TOP_K": 20,
    "SECTION_BOOST_SCORE": 1.5,
}

TRACKED_CONFIG_KEYS = [
    "CHILD_TARGET_CHARS",
    "CHILD_MAX_CHARS",
    "CHUNK_SENTENCE_OVERLAP",
    "PARENT_WINDOW_CHUNKS",
    "PARENT_MAX_CHARS",
    "DENSE_TOP_K",
    "BM25_TOP_K",
    "RRF_TOP_K",
    "RRF_K",
    "DEFAULT_FINAL_K",
    "RERANK_POOL",
    "RERANKER_SCORE_THRESHOLD",
    "RERANKER_GAP_THRESHOLD",
    "SECTION_BOOST_SCORE",
    "EMBED_MODEL",
    "EMBED_QUERY_PREFIX",
    "EMBED_PASSAGE_PREFIX",
    "RERANKER_MODEL",
]

CHUNK_CONTEXT_KEYS = [
    "CHILD_TARGET_CHARS",
    "CHILD_MAX_CHARS",
    "CHUNK_SENTENCE_OVERLAP",
    "PARENT_WINDOW_CHUNKS",
    "PARENT_MAX_CHARS",
]

RETRIEVAL_KEYS = [
    "DENSE_TOP_K",
    "BM25_TOP_K",
    "RRF_TOP_K",
    "RRF_K",
]

SELECTION_KEYS = [
    "RERANK_POOL",
    "DEFAULT_FINAL_K",
    "RERANKER_SCORE_THRESHOLD",
    "RERANKER_GAP_THRESHOLD",
    "SECTION_BOOST_SCORE",
]

CHUNK_CONTEXT_SPACE = {
    "CHILD_TARGET_CHARS": [150, 200, 250],
    "CHILD_MAX_CHARS": [350, 400, 480],
    "CHUNK_SENTENCE_OVERLAP": [1, 2],
    "PARENT_WINDOW_CHUNKS": [1, 2, 3],
    "PARENT_MAX_CHARS": [600, 800],
}

RETRIEVAL_SPACE = {
    "DENSE_TOP_K": [30, 40, 60],
    "BM25_TOP_K": [20, 40, 60],
    "RRF_TOP_K": [15, 20, 30],
    "RRF_K": [40, 60, 80],
}

SELECTION_SPACE = {
    "RERANK_POOL": [10, 15, 20],
    "DEFAULT_FINAL_K": [1, 2, 3],
    "RERANKER_SCORE_THRESHOLD": [-0.5, 0.0, 0.5],
    "RERANKER_GAP_THRESHOLD": [1.5, 2.0, 3.0],
    "SECTION_BOOST_SCORE": [1.0, 1.5, 2.0],
}

SHORTLIST_COUNTS = {
    "chunk_context_joint": 3,
    "retrieval_joint": 3,
    "selection_joint": 5,
}


RUN_CONTEXT = {}
RUN_PROGRESS = {}


# ─────────────────────────────────────────────────────────────────────────────
# Module loading
# ─────────────────────────────────────────────────────────────────────────────
def load_main_module():
    """Import 111511157.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "hw2_main",
        Path(__file__).parent / "111511157.py",
    )
    mod = spec.loader.load_module()  # type: ignore
    return mod


def apply_ablation_baseline(mod):
    """Apply ablation-specific baseline defaults to the loaded main module."""
    for key, value in ABLATION_BASELINE_OVERRIDES.items():
        if hasattr(mod, key):
            setattr(mod, key, value)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def setup_ablation_logging(log_dir: Path, experiment: str):
    """Create one debug log file per ablation command run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_experiment = experiment.replace("/", "_").replace(" ", "_")
    log_path = log_dir / f"ablation_{safe_experiment}_{timestamp}.log"

    logger = build_ablation_logger(log_path, f"ablation.{safe_experiment}.{timestamp}")
    return logger, log_path


def build_ablation_logger(log_path: Path, logger_name: str):
    """Create a logger that writes debug logs to file and warnings to stderr."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    # formatter = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Config / shortlist helpers
# ─────────────────────────────────────────────────────────────────────────────
def snapshot_config(mod, keys=None) -> dict:
    """Capture the current module-level config values used by ablation."""
    keys = keys or TRACKED_CONFIG_KEYS
    return {key: getattr(mod, key) for key in keys if hasattr(mod, key)}


def config_subset(config: dict, keys: list[str]) -> dict:
    """Extract a stable subset of config keys from a full config dict."""
    return {key: config[key] for key in keys if key in config}


def results_score_field(result: dict) -> str:
    """Choose the field used to rank result rows."""
    return "mean_repeat_score" if "mean_repeat_score" in result else "mean_evidence_score"


def ranked_results(results: list[dict]) -> list[dict]:
    """Return results sorted descending by their main score, with rank attached."""
    if not results:
        return []

    score_field = results_score_field(results[0])
    ranked = sorted(results, key=lambda row: -row.get(score_field, row.get("mean_evidence_score", 0.0)))
    out = []
    for rank, row in enumerate(ranked, 1):
        item = dict(row)
        item["rank"] = rank
        out.append(item)
    return out


def joint_shortlist_path(experiment_name: str) -> Path:
    """Path to the cached shortlist used by downstream joint-search stages."""
    return Path(__file__).parent / "outputs" / "joint_search" / f"{experiment_name}.json"


def canonical_results_path(experiment_name: str) -> Path:
    """Canonical saved-results location for an experiment."""
    outputs_dir = Path(__file__).parent / "outputs"
    if experiment_name.startswith("chunk_context_"):
        return outputs_dir / "chunk_context" / f"{experiment_name}.json"
    if experiment_name == "retrieval_joint":
        return outputs_dir / "retrieval" / "retrieval_joint.json"
    return outputs_dir / f"{experiment_name}.json"


def legacy_results_paths(experiment_name: str) -> list[Path]:
    """Backward-compatible result locations from older output layouts."""
    outputs_dir = Path(__file__).parent / "outputs"
    if experiment_name == "chunk_context_joint":
        return [outputs_dir / "ablation" / "chunk_contect" / "chunk_context_joint.json"]
    return []


def default_output_path(experiment_name: str, output_arg: str) -> Path:
    """Resolve the default output path for the current experiment."""
    if output_arg != "ablation_results.json":
        return Path(__file__).parent / "outputs" / output_arg
    if experiment_name in {"chunk_context_joint", "chunk_context_top10", "retrieval_joint"}:
        return canonical_results_path(experiment_name)
    return Path(__file__).parent / "outputs" / output_arg


def save_joint_shortlist(experiment_name: str, results: list[dict], top_n: int, group_keys: list[str]) -> Path:
    """Persist the ranked shortlist for downstream staged joint search."""
    ranked = ranked_results(results)
    out_path = joint_shortlist_path(experiment_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_n": top_n,
        "group_keys": group_keys,
        "results": [
            {key: value for key, value in row.items() if key != "scores"}
            for row in ranked[:top_n]
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def load_joint_shortlist(experiment_name: str) -> tuple[list[dict], Path]:
    """Load a previously saved shortlist for a staged joint-search experiment."""
    path = joint_shortlist_path(experiment_name)
    if not path.exists():
        if experiment_name == "chunk_context_joint":
            source_path = canonical_results_path(experiment_name)
            if not source_path.exists():
                results, source_path = load_saved_experiment_results(experiment_name)
            else:
                with open(source_path, encoding="utf-8") as f:
                    payload = json.load(f)
                results = payload.get(experiment_name, [])
            ranked = ranked_results(results)
            top_n = SHORTLIST_COUNTS.get(experiment_name, 0)
            if len(ranked) < top_n:
                raise RuntimeError(
                    f"Full results for {experiment_name!r} only have {len(ranked)} rows; "
                    f"need at least {top_n} to build a fallback shortlist."
                )
            return ranked[:top_n], source_path
        raise FileNotFoundError(
            f"Missing shortlist for {experiment_name!r}: {path}. "
            f"Run --experiment {experiment_name} first."
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", [])
    if not results:
        raise RuntimeError(
            f"Shortlist file for {experiment_name!r} is empty: {path}. "
            f"Re-run --experiment {experiment_name}."
        )
    return results, path


def load_saved_experiment_results(experiment_name: str) -> tuple[list[dict], Path]:
    """Load full saved results for an experiment from its checkpoint or output JSON."""
    checkpoint_path = experiment_checkpoint_path(experiment_name)
    if checkpoint_path is not None and checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("experiment") == experiment_name and payload.get("results"):
            if not payload.get("is_complete", False):
                raise RuntimeError(
                    f"Saved checkpoint for {experiment_name!r} is incomplete: {checkpoint_path}. "
                    f"Finish --experiment {experiment_name} first."
                )
            return payload["results"], checkpoint_path

    outputs_dir = Path(__file__).parent / "outputs"
    candidate_paths = [canonical_results_path(experiment_name), *legacy_results_paths(experiment_name)]
    output_path = RUN_CONTEXT.get("output_path")
    if output_path is not None and output_path not in candidate_paths:
        candidate_paths.append(output_path)
    candidate_paths.append(outputs_dir / "ablation_results.json")

    for path in candidate_paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            results = payload.get(experiment_name)
            if results:
                return results, path

            if payload.get("experiment") == experiment_name and payload.get("results"):
                return payload["results"], path

    expected_paths = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f"Missing saved results for {experiment_name!r}. Expected one of: {expected_paths}"
    )


def describe_result_score(result: dict) -> float:
    """Score value used for tables and ranking."""
    return result.get("mean_repeat_score", result.get("mean_evidence_score", 0.0))


def sanitize_name(name: str) -> str:
    """Convert a string into a filesystem-friendly filename fragment."""
    safe = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name]
    return "".join(safe).strip("_") or "run"


def build_combo_key(label: str, config: dict) -> str:
    """Stable identifier for a config row used by checkpoint resume."""
    return json.dumps(
        {"label": label, "config": config},
        ensure_ascii=False,
        sort_keys=True,
    )


def json_dump_atomic(path: Path, payload: dict):
    """Atomically write JSON so partial runs never leave a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


def serialize_results(all_results: dict) -> dict:
    """Drop verbose per-paper fields before saving snapshots/results."""
    serializable = {}
    for exp_name, results in all_results.items():
        ranked = ranked_results(results)
        serializable[exp_name] = [
            {key: value for key, value in row.items() if key not in {"scores", "combo_key"}}
            for row in ranked
        ]
    return serializable


def experiment_checkpoint_path(experiment_name: str) -> Path | None:
    """Return the checkpoint path for the current experiment run."""
    checkpoint_root = RUN_CONTEXT.get("checkpoint_root")
    if checkpoint_root is None:
        return None

    sample = RUN_CONTEXT.get("sample", 0)
    sample_tag = "all" if sample == 0 else f"sample{sample}"
    seed_tag = f"seed{RUN_CONTEXT.get('seed', 0)}"
    dataset_tag = f"n{RUN_CONTEXT.get('dataset_size', 0)}"
    stem = f"{sanitize_name(experiment_name)}_{sample_tag}_{seed_tag}_{dataset_tag}"
    return checkpoint_root / f"{stem}.progress.json"


def load_experiment_progress(experiment_name: str) -> dict:
    """Load prior unfinished progress for a resumable experiment."""
    checkpoint_path = experiment_checkpoint_path(experiment_name)
    state = {
        "checkpoint_path": checkpoint_path,
        "loaded": False,
        "was_complete": False,
        "results": [],
    }
    if checkpoint_path is None or RUN_CONTEXT.get("reset_progress"):
        return state
    if not checkpoint_path.exists():
        return state

    with open(checkpoint_path, encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("experiment") != experiment_name:
        return state

    if payload.get("is_complete"):
        state["was_complete"] = True
        return state

    state["loaded"] = True
    state["results"] = payload.get("results", [])
    return state


def save_experiment_progress(
    experiment_name: str,
    results: list[dict],
    total_configs: int,
    completed_configs: int,
):
    """Persist resumable state after each finished configuration."""
    checkpoint_path = experiment_checkpoint_path(experiment_name)
    if checkpoint_path is None:
        return

    payload = {
        "experiment": experiment_name,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_configs": total_configs,
        "completed_configs": completed_configs,
        "is_complete": completed_configs >= total_configs,
        "results": results,
    }
    json_dump_atomic(checkpoint_path, payload)


def save_live_output_snapshot(experiment_name: str, results: list[dict]):
    """Update the main output JSON during a long-running single experiment."""
    if not RUN_CONTEXT.get("enable_live_output"):
        return

    output_path = RUN_CONTEXT.get("output_path")
    if output_path is None:
        return

    json_dump_atomic(output_path, serialize_results({experiment_name: results}))


def run_resumable_experiment(experiment_name: str, combos, mod, dataset, logger):
    """Run a config sweep with per-config persistence and auto-resume."""
    total_configs = len(combos)
    progress_state = load_experiment_progress(experiment_name)
    checkpoint_path = progress_state["checkpoint_path"]
    if checkpoint_path is not None:
        print(f"  Progress file: {checkpoint_path}")

    combo_keys = {}
    for idx, (label, cfg) in enumerate(combos, 1):
        combo_keys[build_combo_key(label, cfg)] = idx

    results = []
    completed_keys = set()
    ignored_rows = 0
    for row in progress_state["results"]:
        combo_key = row.get("combo_key")
        if combo_key in combo_keys and combo_key not in completed_keys:
            results.append(row)
            completed_keys.add(combo_key)
        else:
            ignored_rows += 1

    if progress_state["loaded"]:
        remaining = [idx for key, idx in combo_keys.items() if key not in completed_keys]
        next_idx = remaining[0] if remaining else total_configs
        print(
            f"  Resume       : {len(completed_keys)}/{total_configs} completed, next config #{next_idx}"
        )
        if ignored_rows:
            print(f"  Resume note  : ignored {ignored_rows} stale checkpoint rows")
    elif progress_state["was_complete"]:
        print("  Resume       : previous checkpoint already completed, starting from config #1")
    print()

    processed_this_run = 0
    max_configs_per_run = max(RUN_CONTEXT.get("max_configs_per_run", 0), 0)

    for idx, (label, cfg) in enumerate(combos, 1):
        combo_key = build_combo_key(label, cfg)
        if combo_key in completed_keys:
            continue

        print(f"  [{idx:>3}/{total_configs}] {label}")
        result = run_and_print(mod, dataset, cfg, label, logger)
        result["combo_key"] = combo_key
        results.append(result)
        completed_keys.add(combo_key)
        processed_this_run += 1

        save_experiment_progress(
            experiment_name,
            results,
            total_configs=total_configs,
            completed_configs=len(completed_keys),
        )
        save_live_output_snapshot(experiment_name, results)

        if max_configs_per_run > 0 and processed_this_run >= max_configs_per_run:
            break

    completed_count = len(completed_keys)
    is_complete = completed_count >= total_configs
    save_experiment_progress(
        experiment_name,
        results,
        total_configs=total_configs,
        completed_configs=completed_count,
    )
    save_live_output_snapshot(experiment_name, results)

    next_unfinished = None
    for idx, (label, cfg) in enumerate(combos, 1):
        if build_combo_key(label, cfg) not in completed_keys:
            next_unfinished = idx
            break

    stopped_early = (
        max_configs_per_run > 0
        and processed_this_run >= max_configs_per_run
        and not is_complete
    )
    if stopped_early and next_unfinished is not None:
        print(
            f"  Stop point   : processed {processed_this_run} unfinished configs this run; "
            f"next start will be config #{next_unfinished}"
        )
        print("  Continue     : rerun the same command to resume from that checkpoint")
        print()

    RUN_PROGRESS[experiment_name] = {
        "checkpoint_path": checkpoint_path,
        "completed_configs": completed_count,
        "total_configs": total_configs,
        "is_complete": is_complete,
        "stopped_early": stopped_early,
        "processed_this_run": processed_this_run,
        "next_unfinished": next_unfinished,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(mod, dataset, config_overrides: dict, label: str, logger: logging.Logger) -> dict:
    """Run pipeline with specific config overrides and return evidence scores."""
    original_values = {}
    for key, value in config_overrides.items():
        if not hasattr(mod, key):
            print(f"  ⚠ Warning: {key} not found in module, skipping", file=sys.stderr)
            continue
        original_values[key] = getattr(mod, key)
        setattr(mod, key, value)

    try:
        logger.debug("")
        logger.debug("%s", "=" * 80)
        logger.debug("Config Run: %s", label)
        logger.debug("Overrides: %s", json.dumps(config_overrides, ensure_ascii=False, sort_keys=True))
        logger.debug("%s", "=" * 80)

        api_key = os.environ.get("OPENROUTER_API_KEY", "ollama")
        pipeline = mod.RAGPipeline(
            api_key=api_key,
            logger=logger,
            enable_hyde=True,
            enable_generation=False,
        )

        t0 = time.time()
        results = pipeline.run(dataset)
        elapsed = time.time() - t0

        evaluator = mod.Evaluator(logger=logger)
        eval_result = evaluator.evaluate(results, dataset)

        return {
            "label": label,
            "config": config_overrides,
            "mean_evidence_score": eval_result.get("mean_evidence_score", 0),
            "scores": eval_result.get("scores", []),
            "n": eval_result.get("n", 0),
            "elapsed_sec": round(elapsed, 1),
            "time_sec": round(elapsed, 1),
        }
    finally:
        for key, value in original_values.items():
            setattr(mod, key, value)


def run_experiment_subprocess(
    dataset_path: Path,
    config_overrides: dict,
    label: str,
    log_path: Path,
    experiment: str,
) -> dict:
    """Run a single config in a fresh Python subprocess to avoid model resource leaks."""
    with tempfile.TemporaryDirectory(prefix="ablation_combo_") as tmp_dir:
        result_path = Path(tmp_dir) / "result.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--experiment", experiment,
            "--worker-config-json", json.dumps(config_overrides, ensure_ascii=False, sort_keys=True),
            "--worker-label", label,
            "--worker-dataset-path", str(dataset_path),
            "--worker-result-path", str(result_path),
            "--worker-log-path", str(log_path),
        ]
        completed = subprocess.run(cmd, check=False)

        if completed.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed for {label!r} with exit code {completed.returncode}. "
                f"See {log_path} for details."
            )
        if not result_path.exists():
            raise RuntimeError(
                f"Subprocess for {label!r} finished without writing {result_path}. "
                f"See {log_path} for details."
            )

        with open(result_path, encoding="utf-8") as f:
            return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment definitions (aligned to current 111511157.py constants)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_k(mod, dataset, logger):
    """Test DEFAULT_FINAL_K = 1, 2, 3."""
    print_header("DEFAULT_FINAL_K")
    combos = [(f"K={k}", {"DEFAULT_FINAL_K": k}) for k in [1, 2, 3]]
    print(f"  Total configurations: {len(combos)}\n")
    return run_resumable_experiment("k", combos, mod, dataset, logger)


def experiment_chunk_target(mod, dataset, logger):
    """Test CHILD_TARGET_CHARS = 100, 150, 200, 250, 300, 400."""
    print_header("CHILD_TARGET_CHARS")
    combos = [(f"target={t}", {"CHILD_TARGET_CHARS": t}) for t in [100, 150, 200, 250, 300, 400]]
    print(f"  Total configurations: {len(combos)}\n")
    return run_resumable_experiment("chunk_target", combos, mod, dataset, logger)


def experiment_chunk_max(mod, dataset, logger):
    """Test CHILD_MAX_CHARS = 300, 400, 500, 600."""
    print_header("CHILD_MAX_CHARS")
    combos = [(f"max={m}", {"CHILD_MAX_CHARS": m}) for m in [300, 400, 500, 600]]
    print(f"  Total configurations: {len(combos)}\n")
    return run_resumable_experiment("chunk_max", combos, mod, dataset, logger)


def experiment_overlap(mod, dataset, logger):
    """Test CHUNK_SENTENCE_OVERLAP = 0, 1, 2."""
    print_header("CHUNK_SENTENCE_OVERLAP")
    combos = [(f"overlap={o}", {"CHUNK_SENTENCE_OVERLAP": o}) for o in [0, 1, 2]]
    print(f"  Total configurations: {len(combos)}\n")
    return run_resumable_experiment("overlap", combos, mod, dataset, logger)


def experiment_parent(mod, dataset, logger):
    """Test PARENT_WINDOW_CHUNKS = 1, 2, 3, 4."""
    print_header("PARENT_WINDOW_CHUNKS")
    combos = [(f"parent_win={w}", {"PARENT_WINDOW_CHUNKS": w}) for w in [1, 2, 3, 4]]
    print(f"  Total configurations: {len(combos)}\n")
    return run_resumable_experiment("parent", combos, mod, dataset, logger)


def experiment_threshold(mod, dataset, logger):
    """Test reranker threshold combos."""
    print_header("Reranker Thresholds (score, gap)")
    configs = [
        {"RERANKER_SCORE_THRESHOLD": -2.0, "RERANKER_GAP_THRESHOLD": 10.0},  # very loose
        {"RERANKER_SCORE_THRESHOLD": -1.0, "RERANKER_GAP_THRESHOLD": 5.0},   # loose
        {"RERANKER_SCORE_THRESHOLD": 0.0,  "RERANKER_GAP_THRESHOLD": 2.0},   # current
        {"RERANKER_SCORE_THRESHOLD": 0.5,  "RERANKER_GAP_THRESHOLD": 1.5},   # moderate
        {"RERANKER_SCORE_THRESHOLD": 1.0,  "RERANKER_GAP_THRESHOLD": 1.0},   # aggressive
        {"RERANKER_SCORE_THRESHOLD": 2.0,  "RERANKER_GAP_THRESHOLD": 0.5},   # very aggressive
    ]
    combos = []
    for cfg in configs:
        label = f"score≥{cfg['RERANKER_SCORE_THRESHOLD']},gap≤{cfg['RERANKER_GAP_THRESHOLD']}"
        combos.append((label, cfg))
    print(f"  Total configurations: {len(combos)}\n")
    return run_resumable_experiment("threshold", combos, mod, dataset, logger)


def experiment_section_boost(mod, dataset, logger):
    """Test SECTION_BOOST_SCORE values."""
    print_header("SECTION_BOOST_SCORE")
    combos = [(f"boost={b}", {"SECTION_BOOST_SCORE": b}) for b in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]]
    print(f"  Total configurations: {len(combos)}\n")
    return run_resumable_experiment("section", combos, mod, dataset, logger)


def experiment_rerank_pool(mod, dataset, logger):
    """Test RERANK_POOL sizes."""
    print_header("RERANK_POOL")
    combos = [(f"pool={p}", {"RERANK_POOL": p}) for p in [10, 15, 20, 25, 30]]
    print(f"  Total configurations: {len(combos)}\n")
    return run_resumable_experiment("rerank_pool", combos, mod, dataset, logger)


def experiment_rrf_top_k(mod, dataset, logger):
    """Test RRF_TOP_K values."""
    print_header("RRF_TOP_K")
    combos = [(f"rrf_k={k}", {"RRF_TOP_K": k}) for k in [15, 20, 25, 30, 40]]
    print(f"  Total configurations: {len(combos)}\n")
    return run_resumable_experiment("rrf_top_k", combos, mod, dataset, logger)


def print_joint_ranking(title: str, results: list[dict], top_n: int = 10):
    """Print a compact ranking table for a joint-search experiment."""
    ranked = ranked_results(results)
    if not ranked:
        return

    shown = ranked[:min(top_n, len(ranked))]
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")
    best_score = describe_result_score(ranked[0])
    for row in shown:
        score = describe_result_score(row)
        delta = score - best_score
        delta_str = "" if row["rank"] == 1 else f" ({delta:+.4f})"
        print(f"  {row['rank']:>2}. {row['label']:<48s} {score:.4f}{delta_str}  ({row['time_sec']}s)")


def experiment_chunk_context_joint(mod, dataset, logger):
    """Joint search over chunk geometry and parent context."""
    print_header("Joint Search: Chunk + Parent Context")

    base_config = snapshot_config(mod)
    combos = []
    for target, max_chars, overlap, parent_window, parent_max in itertools.product(
        CHUNK_CONTEXT_SPACE["CHILD_TARGET_CHARS"],
        CHUNK_CONTEXT_SPACE["CHILD_MAX_CHARS"],
        CHUNK_CONTEXT_SPACE["CHUNK_SENTENCE_OVERLAP"],
        CHUNK_CONTEXT_SPACE["PARENT_WINDOW_CHUNKS"],
        CHUNK_CONTEXT_SPACE["PARENT_MAX_CHARS"],
    ):
        if max_chars < target:
            continue
        cfg = dict(base_config)
        cfg.update({
            "CHILD_TARGET_CHARS": target,
            "CHILD_MAX_CHARS": max_chars,
            "CHUNK_SENTENCE_OVERLAP": overlap,
            "PARENT_WINDOW_CHUNKS": parent_window,
            "PARENT_MAX_CHARS": parent_max,
        })
        label = (
            f"target={target},max={max_chars},overlap={overlap},"
            f"parent_win={parent_window},parent_max={parent_max}"
        )
        combos.append((label, cfg))

    print(f"  Total configurations: {len(combos)}\n")

    results = run_resumable_experiment("chunk_context_joint", combos, mod, dataset, logger)

    ranked = ranked_results(results)
    print_joint_ranking("Chunk + Parent Context — Top 10", ranked, top_n=10)
    progress = RUN_PROGRESS.get("chunk_context_joint", {})
    if progress.get("is_complete"):
        shortlist_path = save_joint_shortlist(
            "chunk_context_joint",
            ranked,
            SHORTLIST_COUNTS["chunk_context_joint"],
            CHUNK_CONTEXT_KEYS,
        )
        print(f"\n  Shortlist saved → {shortlist_path}")
    else:
        print("\n  Shortlist skipped → current run is incomplete; finish all configs to refresh shortlist")
    return ranked


def experiment_chunk_context_top10(mod, dataset, logger):
    """Re-run the top-10 chunk/context configs from a completed joint search."""
    print_header("Chunk + Parent Context — Top 10 Rerun")

    source_results, source_path = load_saved_experiment_results("chunk_context_joint")
    source_ranked = ranked_results(source_results)
    if len(source_ranked) < 10:
        raise RuntimeError(
            f"chunk_context_joint only has {len(source_ranked)} saved results. "
            "Need at least 10 completed configs to rerun the top-10."
        )

    print(f"  Source results : {source_path}")
    print("  Re-running     : top 10 configs from chunk_context_joint")

    base_config = snapshot_config(mod)
    combos = []
    source_by_label = {}
    for row in source_ranked[:10]:
        label = f"seed_rank={row['rank']:02d} {row['label']}"
        cfg = dict(base_config)
        cfg.update(row.get("config", {}))
        combos.append((label, cfg))
        source_by_label[label] = row

    print(f"  Total configurations: {len(combos)}\n")

    results = run_resumable_experiment("chunk_context_top10", combos, mod, dataset, logger)
    ranked = ranked_results(results)
    print_joint_ranking("Chunk + Parent Context Rerun — Top 10", ranked, top_n=10)

    print(f"\n{'─' * 80}")
    print("  Top-10 Rerun vs Original")
    print(f"{'─' * 80}")
    for row in ranked:
        seed_row = source_by_label[row["label"]]
        new_score = describe_result_score(row)
        old_score = describe_result_score(seed_row)
        print(
            f"  {row['rank']:>2}. {row['label']:<60s} "
            f"{new_score:.4f} ({new_score - old_score:+.4f} vs seed)"
        )
    return ranked


def experiment_retrieval_joint(mod, dataset, logger):
    """Joint search over dense/BM25/RRF retrieval capacity."""
    print_header("Joint Search: Retrieval Pool")

    chunk_shortlist, chunk_path = load_joint_shortlist("chunk_context_joint")
    best_chunk = chunk_shortlist[0]
    base_config = snapshot_config(mod)
    base_config.update(config_subset(best_chunk["config"], CHUNK_CONTEXT_KEYS))

    print(f"  Using chunk/context shortlist: {chunk_path}")
    print(f"  Seed config: {best_chunk['label']}")

    combos = []
    for dense_k, bm25_k, rrf_top_k, rrf_k in itertools.product(
        RETRIEVAL_SPACE["DENSE_TOP_K"],
        RETRIEVAL_SPACE["BM25_TOP_K"],
        RETRIEVAL_SPACE["RRF_TOP_K"],
        RETRIEVAL_SPACE["RRF_K"],
    ):
        cfg = dict(base_config)
        cfg.update({
            "DENSE_TOP_K": dense_k,
            "BM25_TOP_K": bm25_k,
            "RRF_TOP_K": rrf_top_k,
            "RRF_K": rrf_k,
        })
        label = f"dense={dense_k},bm25={bm25_k},rrf_top={rrf_top_k},rrf_k={rrf_k}"
        combos.append((label, cfg))

    print(f"  Total configurations: {len(combos)}\n")

    results = run_resumable_experiment("retrieval_joint", combos, mod, dataset, logger)

    ranked = ranked_results(results)
    print_joint_ranking("Retrieval Joint Search — Top 10", ranked, top_n=10)
    progress = RUN_PROGRESS.get("retrieval_joint", {})
    if progress.get("is_complete"):
        shortlist_path = save_joint_shortlist(
            "retrieval_joint",
            ranked,
            SHORTLIST_COUNTS["retrieval_joint"],
            RETRIEVAL_KEYS,
        )
        print(f"\n  Shortlist saved → {shortlist_path}")
    else:
        print("\n  Shortlist skipped → current run is incomplete; finish all configs to refresh shortlist")
    return ranked


def experiment_selection_joint(mod, dataset, logger):
    """Joint search over rerank pool, dynamic-K thresholds, and section boost."""
    print_header("Joint Search: Rerank + Final Evidence Selection")

    retrieval_shortlist, retrieval_path = load_joint_shortlist("retrieval_joint")
    best_retrieval = retrieval_shortlist[0]
    base_config = snapshot_config(mod)
    base_config.update(config_subset(best_retrieval["config"], CHUNK_CONTEXT_KEYS))
    base_config.update(config_subset(best_retrieval["config"], RETRIEVAL_KEYS))

    print(f"  Using retrieval shortlist: {retrieval_path}")
    print(f"  Seed config: {best_retrieval['label']}")

    combos = []
    skipped = 0
    for pool, final_k, score_thresh, gap_thresh, boost in itertools.product(
        SELECTION_SPACE["RERANK_POOL"],
        SELECTION_SPACE["DEFAULT_FINAL_K"],
        SELECTION_SPACE["RERANKER_SCORE_THRESHOLD"],
        SELECTION_SPACE["RERANKER_GAP_THRESHOLD"],
        SELECTION_SPACE["SECTION_BOOST_SCORE"],
    ):
        if pool > base_config["RRF_TOP_K"]:
            skipped += 1
            continue
        cfg = dict(base_config)
        cfg.update({
            "RERANK_POOL": pool,
            "DEFAULT_FINAL_K": final_k,
            "RERANKER_SCORE_THRESHOLD": score_thresh,
            "RERANKER_GAP_THRESHOLD": gap_thresh,
            "SECTION_BOOST_SCORE": boost,
        })
        label = (
            f"pool={pool},K={final_k},score={score_thresh},"
            f"gap={gap_thresh},boost={boost}"
        )
        combos.append((label, cfg))

    print(f"  Total configurations: {len(combos)}")
    if skipped:
        print(f"  Skipped invalid configs (pool > RRF_TOP_K={base_config['RRF_TOP_K']}): {skipped}")
    print()

    results = run_resumable_experiment("selection_joint", combos, mod, dataset, logger)

    ranked = ranked_results(results)
    print_joint_ranking("Selection Joint Search — Top 10", ranked, top_n=10)
    progress = RUN_PROGRESS.get("selection_joint", {})
    if progress.get("is_complete"):
        shortlist_path = save_joint_shortlist(
            "selection_joint",
            ranked,
            SHORTLIST_COUNTS["selection_joint"],
            SELECTION_KEYS,
        )
        print(f"\n  Shortlist saved → {shortlist_path}")
    else:
        print("\n  Shortlist skipped → current run is incomplete; finish all configs to refresh shortlist")
    return ranked


def experiment_finalists_joint(mod, dataset, logger):
    """Cross the stage shortlists and re-run finalists on the full public dataset."""
    print_header("Joint Search: Finalists on Full Public Set")

    chunk_shortlist, chunk_path = load_joint_shortlist("chunk_context_joint")
    retrieval_shortlist, retrieval_path = load_joint_shortlist("retrieval_joint")
    selection_shortlist, selection_path = load_joint_shortlist("selection_joint")

    print(f"  Using chunk shortlist     : {chunk_path}")
    print(f"  Using retrieval shortlist : {retrieval_path}")
    print(f"  Using selection shortlist : {selection_path}")

    base_config = snapshot_config(mod)
    finalists = []
    skipped = 0
    for chunk_row, retrieval_row, selection_row in itertools.product(
        chunk_shortlist[:2],
        retrieval_shortlist[:2],
        selection_shortlist[:3],
    ):
        cfg = dict(base_config)
        cfg.update(config_subset(chunk_row["config"], CHUNK_CONTEXT_KEYS))
        cfg.update(config_subset(retrieval_row["config"], RETRIEVAL_KEYS))
        cfg.update(config_subset(selection_row["config"], SELECTION_KEYS))
        if cfg["RERANK_POOL"] > cfg["RRF_TOP_K"]:
            skipped += 1
            continue

        label = (
            f"C{chunk_row['rank']}-R{retrieval_row['rank']}-S{selection_row['rank']} "
            f"(target={cfg['CHILD_TARGET_CHARS']},dense={cfg['DENSE_TOP_K']},"
            f"pool={cfg['RERANK_POOL']},K={cfg['DEFAULT_FINAL_K']})"
        )
        finalists.append((label, cfg))

    print(f"  Candidate finalists: {len(finalists)}")
    if skipped:
        print(f"  Skipped invalid cross-stage configs: {skipped}")
    print("  Repeats per config: 2 (with tie-break rerun for top-2 if needed)\n")

    results = []
    for i, (label, cfg) in enumerate(finalists, 1):
        print(f"  [{i:>2}/{len(finalists)}] {label}")
        repeat_scores = []
        total_elapsed = 0.0
        n_items = 0
        for run_idx in range(1, 3):
            run_label = f"{label} [run {run_idx}/2]"
            result = run_experiment(mod, dataset, cfg, run_label, logger)
            repeat_scores.append(result["mean_evidence_score"])
            total_elapsed += result["elapsed_sec"]
            n_items = result.get("n", n_items)
        mean_score = sum(repeat_scores) / len(repeat_scores)
        final_row = {
            "label": label,
            "config": cfg,
            "repeat_scores": repeat_scores,
            "mean_repeat_score": mean_score,
            "mean_evidence_score": mean_score,
            "n": n_items,
            "elapsed_sec": round(total_elapsed, 1),
            "time_sec": round(total_elapsed, 1),
        }
        results.append(final_row)
        print(f"    mean={mean_score:.4f}  repeats={[round(s, 4) for s in repeat_scores]}  ({total_elapsed:.1f}s)")

    ranked = ranked_results(results)
    if len(ranked) >= 2 and describe_result_score(ranked[0]) - describe_result_score(ranked[1]) < 0.005:
        print("\n  Top-2 gap < 0.005, running 3rd repeat for the current top-2...")
        rerun_labels = {ranked[0]["label"], ranked[1]["label"]}
        updated = []
        for row in ranked:
            item = dict(row)
            if item["label"] in rerun_labels:
                result = run_experiment(mod, dataset, item["config"], f"{item['label']} [run 3/3]", logger)
                item["repeat_scores"] = item["repeat_scores"] + [result["mean_evidence_score"]]
                item["elapsed_sec"] = round(item["elapsed_sec"] + result["elapsed_sec"], 1)
                item["time_sec"] = item["elapsed_sec"]
                item["mean_repeat_score"] = sum(item["repeat_scores"]) / len(item["repeat_scores"])
                item["mean_evidence_score"] = item["mean_repeat_score"]
            updated.append(item)
        ranked = ranked_results(updated)

    print_joint_ranking("Finalists Joint Search — Top 12", ranked, top_n=12)
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# 30 Embedding + Reranker Model Combos (with per-pair tuned parameters)
# ─────────────────────────────────────────────────────────────────────────────

# Embedding models with their required query/passage prefixes
EMBEDDING_CONFIGS = [
    {
        "name": "bge-large-v1.5",
        "EMBED_MODEL": "BAAI/bge-large-en-v1.5",
        "EMBED_QUERY_PREFIX": "Represent this sentence for searching relevant passages: ",
        "EMBED_PASSAGE_PREFIX": "",
    },
    # { 很爛
    #     "name": "bge-m3",
    #     "EMBED_MODEL": "BAAI/bge-m3",
    #     "EMBED_QUERY_PREFIX": "",  # bge-m3 doesn't need prefix
    #     "EMBED_PASSAGE_PREFIX": "",
    # },
    {
        "name": "e5-large-v2",
        "EMBED_MODEL": "intfloat/e5-large-v2",
        "EMBED_QUERY_PREFIX": "query: ",
        "EMBED_PASSAGE_PREFIX": "passage: ",
    },
    # { 很爛
    #     "name": "gte-large",
    #     "EMBED_MODEL": "thenlper/gte-large",
    #     "EMBED_QUERY_PREFIX": "",
    #     "EMBED_PASSAGE_PREFIX": "",
    # },
    # 有問題
    # {
    #     "name": "mpnet-base-v2",
    #     "EMBED_MODEL": "sentence-transformers/all-mpnet-base-v2",
    #     "EMBED_QUERY_PREFIX": "",
    #     "EMBED_PASSAGE_PREFIX": "",
    # },
    {
        "name": "nomic-v1.5",
        "EMBED_MODEL": "nomic-ai/nomic-embed-text-v1.5",
        "EMBED_QUERY_PREFIX": "search_query: ",
        "EMBED_PASSAGE_PREFIX": "search_document: ",
    },
    {
        "name": "arctic-embed-l-v2",
        "EMBED_MODEL": "Snowflake/snowflake-arctic-embed-l-v2.0",
        # Arctic v2 uses explicit query/passage prefixes for asymmetric retrieval
        "EMBED_QUERY_PREFIX": "Represent this sentence for searching relevant passages: ",
        "EMBED_PASSAGE_PREFIX": "Represent this sentence for retrieval: ",
    },
    {
        "name": "mxbai-embed-large",
        "EMBED_MODEL": "mixedbread-ai/mxbai-embed-large-v1",
        # mxbai-embed uses an explicit query prefix (CLS pooling, no passage prefix needed)
        "EMBED_QUERY_PREFIX": "Represent this sentence for searching relevant passages: ",
        "EMBED_PASSAGE_PREFIX": "",
    },
]

# Reranker models with model-family-specific score calibration
# BGE rerankers output raw logits (range ~ -10 to +10), threshold around 0
# ms-marco MiniLM rerankers output sigmoid-like scores (range ~ 0 to 1), threshold around 0.3-0.5
RERANKER_CONFIGS = [
    {
        "name": "bge-reranker-v2-m3",
        "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
        "score_range": "logit",  # raw logits, can be negative
    },
    {
        "name": "bge-reranker-large",
        "RERANKER_MODEL": "BAAI/bge-reranker-large",
        "score_range": "logit",
    },
    {
        "name": "bge-reranker-base",
        "RERANKER_MODEL": "BAAI/bge-reranker-base",
        "score_range": "logit",
    },
    {
        "name": "ms-marco-MiniLM-L12",
        "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "score_range": "sigmoid",  # outputs ~0-12 logits but calibrated differently
    },
    # {
    #     "name": "ms-marco-MiniLM-L6",
    #     "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    #     "score_range": "sigmoid",
    # },
    {
        "name": "mxbai-rerank-large",
        "RERANKER_MODEL": "mixedbread-ai/mxbai-rerank-large-v1",
        # mxbai-rerank outputs sigmoid-calibrated scores in [0, 1]
        "score_range": "sigmoid",
    },
]

TOP_10_MODEL_COMBO_NAMES = [
    ("bge-large-v1.5", "bge-reranker-base"),
    ("arctic-embed-l-v2", "mxbai-rerank-large"),
    ("arctic-embed-l-v2", "bge-reranker-v2-m3"),
    ("mxbai-embed-large", "bge-reranker-base"),
    ("e5-large-v2", "mxbai-rerank-large"),
    ("arctic-embed-l-v2", "bge-reranker-base"),
    ("nomic-v1.5", "bge-reranker-base"),
    ("e5-large-v2", "bge-reranker-base"),
    ("mxbai-embed-large", "mxbai-rerank-large"),
    ("e5-large-v2", "bge-reranker-v2-m3"),
]

# Per-pair tuned parameters: each (embed, reranker) pair gets parameters
# optimized for that specific combination's characteristics.
#
# Design rationale:
# - Stronger embedders (bge-large, bge-m3, e5-large) retrieve better candidates
#   → can use smaller RERANK_POOL (15) and tighter thresholds
# - Weaker embedders (mpnet, gte-large) need larger RERANK_POOL (20-25) to
#   compensate with reranker quality
# - BGE rerankers use logit scores → RERANKER_SCORE_THRESHOLD ~ 0.0
# - ms-marco rerankers use different scale → threshold ~ -2.0 (their logits are shifted)
# - Strong rerankers (v2-m3, large) → can be more aggressive with K=1
# - Weaker rerankers → safer with K=2
# - CHILD_TARGET_CHARS tuned per embedding model's sweet spot for semantic matching
# - SECTION_BOOST_SCORE higher for weaker embedders to help routing

def _build_pair_config(embed_cfg: dict, reranker_cfg: dict) -> dict:
    """Build a full config dict for a specific (embedding, reranker) pair,
    with parameters tuned to the pair's characteristics."""

    config = {
        "EMBED_MODEL": embed_cfg["EMBED_MODEL"],
        "EMBED_QUERY_PREFIX": embed_cfg["EMBED_QUERY_PREFIX"],
        "EMBED_PASSAGE_PREFIX": embed_cfg["EMBED_PASSAGE_PREFIX"],
        "RERANKER_MODEL": reranker_cfg["RERANKER_MODEL"],
    }

    embed_name = embed_cfg["name"]
    reranker_name = reranker_cfg["name"]
    is_bge_reranker = reranker_cfg["score_range"] == "logit"

    # ── Reranker score thresholds (model-family dependent) ──
    if is_bge_reranker:
        # BGE rerankers output raw logits (range ~ -10 to +10)
        config["RERANKER_SCORE_THRESHOLD"] = 0.0
        config["RERANKER_GAP_THRESHOLD"] = 2.0
    elif reranker_name == "mxbai-rerank-large":
        # mxbai-rerank outputs well-calibrated sigmoid scores in [0, 1]
        # Score of 0.01 is a reasonable floor; gap threshold relative to [0,1] range
        config["RERANKER_SCORE_THRESHOLD"] = 0.01
        config["RERANKER_GAP_THRESHOLD"] = 0.3
    else:
        # ms-marco cross-encoders have different score distribution
        config["RERANKER_SCORE_THRESHOLD"] = -2.0
        config["RERANKER_GAP_THRESHOLD"] = 3.0

    # ── Embedding-model-specific parameters ──
    if embed_name == "bge-large-v1.5":
        # Strong 1024-dim embedder, good at semantic matching
        config["CHILD_TARGET_CHARS"] = 200
        config["CHILD_MAX_CHARS"] = 400
        config["RERANK_POOL"] = 15
        config["RRF_TOP_K"] = 20
        config["DEFAULT_FINAL_K"] = 2
        config["SECTION_BOOST_SCORE"] = 1.5
        config["CHUNK_SENTENCE_OVERLAP"] = 1

    elif embed_name == "bge-m3":
        # Multi-lingual, 1024-dim, slightly different retrieval profile
        config["CHILD_TARGET_CHARS"] = 250
        config["CHILD_MAX_CHARS"] = 500
        config["RERANK_POOL"] = 20
        config["RRF_TOP_K"] = 25
        config["DEFAULT_FINAL_K"] = 2
        config["SECTION_BOOST_SCORE"] = 1.0
        config["CHUNK_SENTENCE_OVERLAP"] = 1

    elif embed_name == "e5-large-v2":
        # Strong 1024-dim with explicit query/passage prefixes
        config["CHILD_TARGET_CHARS"] = 200
        config["CHILD_MAX_CHARS"] = 400
        config["RERANK_POOL"] = 15
        config["RRF_TOP_K"] = 20
        config["DEFAULT_FINAL_K"] = 1
        config["SECTION_BOOST_SCORE"] = 1.5
        config["CHUNK_SENTENCE_OVERLAP"] = 1

    elif embed_name == "gte-large":
        # 1024-dim but slightly weaker on scientific text
        config["CHILD_TARGET_CHARS"] = 200
        config["CHILD_MAX_CHARS"] = 450
        config["RERANK_POOL"] = 20
        config["RRF_TOP_K"] = 25
        config["DEFAULT_FINAL_K"] = 2
        config["SECTION_BOOST_SCORE"] = 2.0
        config["CHUNK_SENTENCE_OVERLAP"] = 1

    elif embed_name == "mpnet-base-v2":
        # 768-dim, general-purpose, needs more reranker help
        config["CHILD_TARGET_CHARS"] = 250
        config["CHILD_MAX_CHARS"] = 500
        config["RERANK_POOL"] = 25
        config["RRF_TOP_K"] = 30
        config["DEFAULT_FINAL_K"] = 2
        config["SECTION_BOOST_SCORE"] = 2.0
        config["CHUNK_SENTENCE_OVERLAP"] = 1

    elif embed_name == "nomic-v1.5":
        # 768-dim with Matryoshka, good prefix system
        config["CHILD_TARGET_CHARS"] = 200
        config["CHILD_MAX_CHARS"] = 400
        config["RERANK_POOL"] = 20
        config["RRF_TOP_K"] = 25
        config["DEFAULT_FINAL_K"] = 2
        config["SECTION_BOOST_SCORE"] = 1.5
        config["CHUNK_SENTENCE_OVERLAP"] = 1

    elif embed_name == "arctic-embed-l-v2":
        # Snowflake Arctic Embed L v2.0 — 1024-dim, strong scientific retrieval
        # Trained on diverse data with Matryoshka representation learning
        # Supports up to 8192 tokens; shorter chunks keep embeddings focused
        config["CHILD_TARGET_CHARS"] = 200
        config["CHILD_MAX_CHARS"] = 400
        config["RERANK_POOL"] = 15
        config["RRF_TOP_K"] = 20
        config["DEFAULT_FINAL_K"] = 1
        config["SECTION_BOOST_SCORE"] = 1.5
        config["CHUNK_SENTENCE_OVERLAP"] = 1

    elif embed_name == "mxbai-embed-large":
        # mixedbread mxbai-embed-large-v1 — 1024-dim, top MTEB performer
        # Uses AnglE loss + CLS pooling; strong on semantic similarity tasks
        # Good at scientific text → can use smaller pool and aggressive K
        config["CHILD_TARGET_CHARS"] = 200
        config["CHILD_MAX_CHARS"] = 400
        config["RERANK_POOL"] = 15
        config["RRF_TOP_K"] = 20
        config["DEFAULT_FINAL_K"] = 1
        config["SECTION_BOOST_SCORE"] = 1.5
        config["CHUNK_SENTENCE_OVERLAP"] = 1

    # ── Reranker-specific adjustments (override some embedding defaults) ──
    if reranker_name == "bge-reranker-v2-m3":
        # Strongest reranker — can trust its ranking, use K=1 more aggressively
        if config["DEFAULT_FINAL_K"] == 2 and embed_name in ("bge-large-v1.5", "e5-large-v2"):
            config["DEFAULT_FINAL_K"] = 1
    elif reranker_name == "bge-reranker-large":
        # Second strongest — slightly less aggressive
        pass
    elif reranker_name == "bge-reranker-base":
        # Weaker — bump pool size to give it more candidates
        config["RERANK_POOL"] = min(config["RERANK_POOL"] + 5, 30)
    elif reranker_name in ("ms-marco-MiniLM-L12", "ms-marco-MiniLM-L6"):
        # ms-marco models are faster but less accurate on scientific text
        # Give them more candidates and be less aggressive with K
        config["RERANK_POOL"] = min(config["RERANK_POOL"] + 5, 30)
        config["DEFAULT_FINAL_K"] = max(config["DEFAULT_FINAL_K"], 2)
    elif reranker_name == "mxbai-rerank-large":
        # mixedbread mxbai-rerank-large-v1 — strong cross-encoder reranker
        # Sigmoid-calibrated scores in [0, 1]; well-calibrated confidence
        # Performs on par with bge-reranker-v2-m3 on MTEB retrieval benchmarks
        # Confident enough to use K=1 with strong embedders
        if embed_name in ("bge-large-v1.5", "e5-large-v2", "arctic-embed-l-v2", "mxbai-embed-large"):
            config["DEFAULT_FINAL_K"] = 1

    return config


def _run_model_combo_experiment(mod, dataset, logger, combos, experiment_name: str, title: str):
    """Run a set of embedding + reranker combinations."""
    print_header(title)
    print(f"  Total configurations: {len(combos)}\n")

    # Print parameter legend
    # print("  Parameters per pair: EMBED_MODEL, RERANKER_MODEL, CHILD_TARGET_CHARS,")
    # print("    CHILD_MAX_CHARS, RERANK_POOL, RRF_TOP_K, DEFAULT_FINAL_K,")
    # print("    SECTION_BOOST_SCORE, RERANKER_SCORE_THRESHOLD, RERANKER_GAP_THRESHOLD,")
    # print("    CHUNK_SENTENCE_OVERLAP\n")

    results = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="ablation_model_combos_") as tmp_dir:
        dataset_path = Path(tmp_dir) / "dataset_snapshot.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False)

        log_path = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_path = Path(handler.baseFilename)
                break
        if log_path is None:
            raise RuntimeError("Ablation logger is missing a FileHandler; cannot launch subprocess combos.")

        for i, (embed_cfg, reranker_cfg) in enumerate(combos, 1):
            config = _build_pair_config(embed_cfg, reranker_cfg)
            label = f"{embed_cfg['name']} + {reranker_cfg['name']}"

            # Print full config for this pair
            print(f"\n  [{i:>2}/{len(combos)}] ⏳ {label}")
            print(f"    Config: K={config['DEFAULT_FINAL_K']}, "
                  f"target={config['CHILD_TARGET_CHARS']}, "
                  f"max={config['CHILD_MAX_CHARS']}, "
                  f"pool={config['RERANK_POOL']}, "
                  f"rrf_k={config['RRF_TOP_K']}, "
                  f"boost={config['SECTION_BOOST_SCORE']}, "
                  f"thr={config['RERANKER_SCORE_THRESHOLD']}, "
                  f"gap={config['RERANKER_GAP_THRESHOLD']}, "
                  f"overlap={config['CHUNK_SENTENCE_OVERLAP']}")

            logger.debug("Dispatching subprocess for %s", label)
            try:
                r = run_experiment_subprocess(
                    dataset_path=dataset_path,
                    config_overrides=config,
                    label=label,
                    log_path=log_path,
                    experiment=experiment_name,
                )
            except Exception as exc:
                err_msg = str(exc)
                logger.error("Model combo failed: %s", label)
                logger.error("Failure detail: %s", err_msg)
                print(f"    {label:<55s} FAILED  ({err_msg})")
                failures.append({
                    "label": label,
                    "config": config,
                    "error": err_msg,
                })
                continue

            bar = "█" * int(r["mean_evidence_score"] * 40)
            print(f"    {label:<55s} {r['mean_evidence_score']:.4f}  {bar}  ({r['time_sec']}s)")
            results.append(r)

    # ── Detailed ranking table ──
    print(f"\n{'─' * 80}")
    print("  MODEL COMBOS — DETAILED RANKING")
    print(f"{'─' * 80}")
    ranked = sorted(results, key=lambda x: -x["mean_evidence_score"])
    best_score = ranked[0]["mean_evidence_score"] if ranked else 0
    for rank, r in enumerate(ranked, 1):
        score = r["mean_evidence_score"]
        delta = score - best_score
        cfg = r["config"]
        print(f"  {rank:>2}. {r['label']:<45s} {score:.4f}"
              f" ({delta:+.4f})"
              f"  K={cfg.get('DEFAULT_FINAL_K','?')}"
              f" pool={cfg.get('RERANK_POOL','?')}"
              f" target={cfg.get('CHILD_TARGET_CHARS','?')}"
              f" thr={cfg.get('RERANKER_SCORE_THRESHOLD','?')}"
              f"  ({r['time_sec']}s)")

    if failures:
        print(f"\n{'─' * 80}")
        print("  FAILED MODEL COMBOS")
        print(f"{'─' * 80}")
        for item in failures:
            print(f"  - {item['label']}: {item['error']}")
        logger.error("Failed model combos: %s", json.dumps(failures, ensure_ascii=False))

    # ── Best pair summary ──
    if ranked:
        best = ranked[0]
        print(f"\n{'═' * 80}")
        print(f"  ★ BEST MODEL PAIR: {best['label']}")
        print(f"  ★ EVIDENCE SCORE:  {best['mean_evidence_score']:.4f}")
        print(f"  ★ FULL CONFIG:")
        for k, v in sorted(best["config"].items()):
            print(f"      {k} = {v!r}")
        print(f"{'═' * 80}")

    return results


def experiment_model_combos(mod, dataset, logger):
    """Test all embedding + reranker model combinations."""
    combos = list(itertools.product(EMBEDDING_CONFIGS, RERANKER_CONFIGS))
    title = f"Model Combos: {len(EMBEDDING_CONFIGS)} Embeddings × {len(RERANKER_CONFIGS)} Rerankers (per-pair tuned)"
    return _run_model_combo_experiment(mod, dataset, logger, combos, "model_combos", title)


def experiment_model_combos_top10(mod, dataset, logger):
    """Run only the selected top-10 embedding + reranker combinations."""
    embed_map = {cfg["name"]: cfg for cfg in EMBEDDING_CONFIGS}
    reranker_map = {cfg["name"]: cfg for cfg in RERANKER_CONFIGS}
    combos = []
    for embed_name, reranker_name in TOP_10_MODEL_COMBO_NAMES:
        combos.append((embed_map[embed_name], reranker_map[reranker_name]))
    return _run_model_combo_experiment(
        mod,
        dataset,
        logger,
        combos,
        "model_combos_top10",
        "Model Combos Top 10 (selected from previous ranking)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────────────────

def experiment_grid(mod, dataset, logger):
    """Grid search over highest-impact parameters.

    K × CHILD_TARGET_CHARS × RERANK_POOL
    Total: 3 × 4 × 3 = 36 configurations
    """
    print_header("Grid Search: K × chunk_target × rerank_pool")

    k_vals = [1, 2, 3]
    target_vals = [150, 200, 250, 300]
    pool_vals = [15, 20, 25]

    combos = list(itertools.product(k_vals, target_vals, pool_vals))
    print(f"  Total configurations: {len(combos)}\n")

    config_rows = []
    for k, t, p in combos:
        cfg = {
            "DEFAULT_FINAL_K": k,
            "CHILD_TARGET_CHARS": t,
            "RERANK_POOL": p,
        }
        label = f"K={k},target={t},pool={p}"
        config_rows.append((label, cfg))
    return run_resumable_experiment("grid", config_rows, mod, dataset, logger)


def experiment_grid_extended(mod, dataset, logger):
    """Extended grid: K × chunk_target × overlap × threshold.

    Total: 2 × 3 × 2 × 2 = 24 configurations
    """
    print_header("Extended Grid: K × chunk_target × overlap × threshold")

    combos = list(itertools.product(
        [1, 2],             # DEFAULT_FINAL_K
        [150, 200, 300],    # CHILD_TARGET_CHARS
        [0, 1],             # CHUNK_SENTENCE_OVERLAP
        [0.0, 0.5],         # RERANKER_SCORE_THRESHOLD
    ))
    print(f"  Total configurations: {len(combos)}\n")

    config_rows = []
    for k, t, o, th in combos:
        cfg = {
            "DEFAULT_FINAL_K": k,
            "CHILD_TARGET_CHARS": t,
            "CHUNK_SENTENCE_OVERLAP": o,
            "RERANKER_SCORE_THRESHOLD": th,
        }
        label = f"K={k},target={t},overlap={o},thr={th}"
        config_rows.append((label, cfg))
    return run_resumable_experiment("grid_ext", config_rows, mod, dataset, logger)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────
def print_header(name: str):
    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'=' * 70}")


def run_and_print(mod, dataset, config: dict, label: str, logger: logging.Logger) -> dict:
    """Run experiment and print result inline."""
    r = run_experiment(mod, dataset, config, label, logger)
    bar = "█" * int(r["mean_evidence_score"] * 40)
    print(f"    {label:<55s} {r['mean_evidence_score']:.4f}  {bar}  ({r['time_sec']}s)", flush=True)
    return r


def print_summary(all_results: dict):
    """Print ranked comparison table for all experiments."""
    print(f"\n{'=' * 70}")
    print("  ABLATION STUDY SUMMARY")
    print(f"{'=' * 70}")

    for exp_name, results in all_results.items():
        print(f"\n  ── {exp_name} ──")
        results_sorted = ranked_results(results)
        best_score = describe_result_score(results_sorted[0]) if results_sorted else 0
        for rank, r in enumerate(results_sorted, 1):
            score = describe_result_score(r)
            delta = score - best_score
            bar = "█" * int(score * 40)
            marker = " ★" if rank == 1 else ""
            delta_str = f" ({delta:+.4f})" if rank > 1 else ""
            print(f"    {rank}. {r['label']:<52s} {score:.4f}{delta_str}  {bar}{marker}")

    # Global best
    all_flat = [r for results in all_results.values() for r in results]
    if all_flat:
        best = max(all_flat, key=describe_result_score)
        print(f"\n  {'─' * 60}")
        print(f"  Best overall: {best['label']}")
        print(f"  Score: {describe_result_score(best):.4f}")
        print(f"  Config: {json.dumps(best['config'], indent=4)}")
        print(f"  {'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal worker mode
# ─────────────────────────────────────────────────────────────────────────────
def run_worker_mode(args) -> int:
    """Execute a single config in a subprocess worker."""
    if not all([
        args.worker_config_json,
        args.worker_label,
        args.worker_dataset_path,
        args.worker_result_path,
        args.worker_log_path,
    ]):
        raise ValueError("Worker mode requires config, label, dataset path, result path, and log path.")

    logger = build_ablation_logger(Path(args.worker_log_path), f"ablation.worker.{os.getpid()}")
    mod = load_main_module()
    apply_ablation_baseline(mod)

    with open(args.worker_dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    config = json.loads(args.worker_config_json)
    result = run_experiment(mod, dataset, config, args.worker_label, logger)

    with open(args.worker_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = {
    "k":             experiment_k,
    "chunk_target":  experiment_chunk_target,
    "chunk_max":     experiment_chunk_max,
    "overlap":       experiment_overlap,
    "parent":        experiment_parent,
    "threshold":     experiment_threshold,
    "section":       experiment_section_boost,
    "rerank_pool":   experiment_rerank_pool,
    "rrf_top_k":     experiment_rrf_top_k,
    "chunk_context_joint": experiment_chunk_context_joint,
    "chunk_context_top10": experiment_chunk_context_top10,
    "retrieval_joint": experiment_retrieval_joint,
    "selection_joint": experiment_selection_joint,
    "finalists_joint": experiment_finalists_joint,
    "model_combos":  experiment_model_combos,
    "model_combos_top10": experiment_model_combos_top10,
    "grid":          experiment_grid,
    "grid_ext":      experiment_grid_extended,
}

# "all" runs these (skip slow model comparisons)
ALL_FAST = ["k", "chunk_target", "chunk_max", "overlap", "parent",
            "threshold", "section", "rerank_pool", "rrf_top_k"]


def main():
    parser = argparse.ArgumentParser(
        description="HW2 Ablation Study — find optimal RAG config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Which experiment to run (default: all fast experiments)",
    )
    parser.add_argument("--sample", type=int, default=20,
                        help="Number of papers to sample (0=all, default=20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="ablation_results.json",
                        help="Output JSON path")
    parser.add_argument(
        "--max-configs-per-run",
        type=int,
        default=0,
        help="Run at most this many unfinished configs before stopping (0=run all)",
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Ignore saved checkpoint progress and restart the selected experiment",
    )
    parser.add_argument("--worker-config-json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-label", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-dataset-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-log-path", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_config_json is not None:
        sys.exit(run_worker_mode(args))

    if args.max_configs_per_run < 0:
        parser.error("--max-configs-per-run must be >= 0")
    if args.max_configs_per_run > 0 and args.experiment == "all":
        parser.error("--max-configs-per-run only supports a single experiment, not --experiment all")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    # Load dataset
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "datasets" / "public_dataset.json"
    if not dataset_path.exists():
        dataset_path = script_dir / "public_dataset.json"
    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}")

    with open(dataset_path) as f:
        full_dataset = json.load(f)

    dataset = list(full_dataset)
    if args.sample > 0 and args.sample < len(full_dataset):
        random.seed(args.seed)
        dataset = random.sample(full_dataset, args.sample)

    run_dataset = full_dataset if args.experiment == "finalists_joint" else dataset
    out_path = default_output_path(args.experiment, args.output)
    RUN_CONTEXT.clear()
    RUN_CONTEXT.update({
        "output_path": out_path,
        "checkpoint_root": script_dir / "outputs" / "ablation_progress",
        "sample": 0 if len(run_dataset) == len(full_dataset) else args.sample,
        "seed": args.seed,
        "dataset_size": len(run_dataset),
        "max_configs_per_run": args.max_configs_per_run,
        "reset_progress": args.reset_progress,
        "enable_live_output": args.experiment != "all",
    })
    RUN_PROGRESS.clear()

    print(f"{'#' * 70}")
    print(f"# Ablation Study — {len(run_dataset)} papers, experiment={args.experiment}")
    print(f"{'#' * 70}")
    if args.experiment == "finalists_joint" and len(dataset) != len(full_dataset):
        print("  Note      : finalists_joint ignores --sample and runs on all public papers")
    if args.max_configs_per_run > 0:
        print(f"  Stop point: run at most {args.max_configs_per_run} unfinished configs this time")
    if args.reset_progress:
        print("  Resume    : ignoring saved checkpoint and starting fresh")

    ablation_logger, log_path = setup_ablation_logging(script_dir / "logs" / "ablation", args.experiment)
    print(f"  Log file  : {log_path}")
    print("  Ablation  : HyDE enabled, final answer generation disabled")
    ablation_logger.debug("%s", "#" * 70)
    ablation_logger.debug("Ablation Study — %d papers, experiment=%s", len(run_dataset), args.experiment)
    ablation_logger.debug("Log file: %s", log_path)
    ablation_logger.debug("Ablation mode: enable_hyde=True, enable_generation=False")
    ablation_logger.debug("%s", "#" * 70)

    # Load main module
    mod = load_main_module()
    apply_ablation_baseline(mod)

    # Print current baseline config
    print(f"\n  Current config:")
    baseline_config = {}
    for const in TRACKED_CONFIG_KEYS:
        value = getattr(mod, const, "?")
        baseline_config[const] = value
        print(f"    {const} = {value}")
    ablation_logger.debug("Baseline config: %s", json.dumps(baseline_config, ensure_ascii=False, sort_keys=True))

    # Run experiments
    all_results = {}
    if args.experiment == "all":
        for exp_name in ALL_FAST:
            all_results[exp_name] = EXPERIMENTS[exp_name](mod, dataset, ablation_logger)
    else:
        exp_name = args.experiment
        all_results[exp_name] = EXPERIMENTS[exp_name](mod, run_dataset, ablation_logger)

    # Summary
    print_summary(all_results)

    # Save results (drop per-paper scores for readability)
    serializable = serialize_results(all_results)
    json_dump_atomic(out_path, serializable)
    print(f"\nResults saved → {out_path}")
    if args.experiment in RUN_PROGRESS:
        progress = RUN_PROGRESS[args.experiment]
        checkpoint_path = progress.get("checkpoint_path")
        if checkpoint_path is not None:
            print(f"Progress saved → {checkpoint_path}")
        if not progress.get("is_complete"):
            print(
                f"Next resume → config #{progress.get('next_unfinished')} "
                f"({progress.get('completed_configs')}/{progress.get('total_configs')} completed)"
            )
    print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()
