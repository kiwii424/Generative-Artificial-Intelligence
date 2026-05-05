"""
HW2 RAG — Self-Scoring Script (Public Dataset)

Compare your pipeline output against the public dataset ground truth.

  Evidence Score  — ROUGE-L F-measure (identical algorithm to TA grading)
  Correctness     — Lightweight LLM judge using the configured API model
                    ⚠ This is an approximation: TA grading uses a separate, more
                    capable model with a different evaluation procedure.

Usage:
  python score_public.py results.json
  python score_public.py outputs/public.json
  python score_public.py outputs/public.json --model meta-llama/llama-3.2-3b-instruct:free
  python score_public.py outputs/public.json --base-url http://localhost:8091/v1 --api-key abc

Input JSON format (same as submission):
  [{"title": "...", "answer": "...", "evidence": [...]}, ...]

Output:
  - Console: per-paper scores and running averages
  - results_score.json: full breakdown
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rouge_score import rouge_scorer

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Self-score HW2 results on the public dataset")
parser.add_argument("results",      type=str,                                        help="Path to your results JSON.")
parser.add_argument("--provider",   type=str, default=os.getenv("LLM_PROVIDER", "openrouter"), help="LLM provider: openrouter, ollama, or custom. Default: openrouter")
parser.add_argument("--base-url",   type=str, default=None,                          help="OpenAI-compatible API base URL. Default: provider/env setting")
parser.add_argument("--api-key",    type=str, default=None,                          help="API key. Default: LLM_API_KEY or OPENROUTER_API_KEY from .env")
parser.add_argument("--host",       type=str, default=None,                          help="Backward-compatible local API host; builds http://HOST:PORT/v1")
parser.add_argument("--port",       type=int, default=None,                          help="Backward-compatible local API port; builds http://HOST:PORT/v1")
parser.add_argument("--model",      type=str, default=None,                          help="LLM model name. Default: LLM_MODEL or provider default")
parser.add_argument("--dataset",    type=str, default="datasets/public_dataset.json",help="Path to the public dataset JSON.")
parser.add_argument("--times",      type=int, default=5,                             help="Judge runs per paper (majority vote). Default: 5")
parser.add_argument("--temperature",type=float, default=0.0,                          help="Judge temperature. Default: 0.0")
args = parser.parse_args()

LLM_PROVIDER = args.provider.strip().lower()
if LLM_PROVIDER not in {"openrouter", "ollama", "custom"}:
  sys.exit("--provider must be one of: openrouter, ollama, custom")

def _default_base_url(provider: str) -> str:
  if args.host or args.port:
    return f"http://{args.host or 'localhost'}:{args.port or 8091}/v1"
  if args.base_url:
    return args.base_url
  if os.getenv("LLM_BASE_URL"):
    return os.environ["LLM_BASE_URL"]
  if provider == "ollama":
    return "http://127.0.0.1:11434/v1"
  if provider == "custom":
    sys.exit("--provider custom requires --base-url or LLM_BASE_URL.")
  return "https://openrouter.ai/api/v1"

def _default_model(provider: str) -> str:
  if args.model:
    return args.model
  if os.getenv("LLM_MODEL"):
    return os.environ["LLM_MODEL"]
  if provider == "ollama":
    return "llama3.2:3b"
  return "meta-llama/llama-3.2-3b-instruct:free"

def _looks_local_api(base_url: str) -> bool:
  return any(host in base_url for host in ("localhost", "127.0.0.1", "0.0.0.0"))

LLM_BASE_URL = _default_base_url(LLM_PROVIDER)
LLM_MODEL = _default_model(LLM_PROVIDER)
LLM_API_KEY = args.api_key or os.getenv("LLM_API_KEY")
if not LLM_API_KEY and LLM_PROVIDER == "openrouter":
  LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not LLM_API_KEY and (LLM_PROVIDER == "ollama" or _looks_local_api(LLM_BASE_URL)):
  LLM_API_KEY = "ollama"
if not LLM_API_KEY:
  sys.exit("API key not found. Set LLM_API_KEY or OPENROUTER_API_KEY in .env, or pass --api-key.")

# ──────────────────────────────────────────────────────────────────────────────
# Evidence Score — ROUGE-L (identical to TA grading)
# ──────────────────────────────────────────────────────────────────────────────
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def calc_evidence_score(retrieved: list[str], golden: list[str]) -> float:
  if not golden:
    return 1.0
  if not retrieved:
    return 0.0
  return sum(
    _rouge.score_multi(targets=golden, prediction=c)["rougeL"].fmeasure
    for c in retrieved
  ) / len(retrieved)

# ──────────────────────────────────────────────────────────────────────────────
# LLM Judge
# ──────────────────────────────────────────────────────────────────────────────
_PROMPT_JUDGEMENT = """\
You are an expert evaluator. Given a document, a question, a ground truth answer, and a model prediction, decide if the prediction is correct.

Evaluation rules (apply in order, stop at the first match):
a) Ground Truth is always correct by definition.
b) If the Prediction expresses uncertainty (e.g. "I don't know", "I'm not sure"), score = 0.
c) If the Prediction exactly matches the Ground Truth (case-insensitive), score = 1.
d) If the Ground Truth is a number, score = 1 only if the Prediction states the same number.
e) If the Prediction is self-contradictory or does not address the question, score = 0.
f) If the Prediction is semantically equivalent to the Ground Truth or is a correct concise summary of it, score = 1.
g) If the Ground Truth lists multiple items, score = 1 only if the Prediction covers all key items (exact wording not required; semantic match is acceptable).
h) Otherwise, score = 0.

IMPORTANT: The score MUST be exactly 0 or 1. No other value is valid.
Return only the single digit 0 or 1.
"""
_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

def _run_judge(query: dict[str, str]) -> str:
  user_prompt = (
    f"document: {query['document']}\n"
    f"question: {query['question']}\n"
    f"Ground Truth: {query['answer']}\n"
    f"Prediction: {query['prediction']}\n\n"
    "Score:"
  )
  resp = _client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
      {"role": "system", "content": _PROMPT_JUDGEMENT},
      {"role": "user", "content": user_prompt},
    ],
    temperature=args.temperature,
    max_tokens=4,
  )
  return (resp.choices[0].message.content or "").strip()

_IDK_RE = re.compile(
  r"i (don'?t|do not|cannot|can'?t) know"
  r"|i'?m not sure"
  r"|not (enough|sufficient) (information|context|detail)"
  r"|the (context|document|passage|text) does(n'?t| not) (contain|mention|provide|have|include|say)",
  re.IGNORECASE,
)

def _extract_score(text: str) -> float:
  """Extract the judge's 0/1 answer from short API output."""
  m = re.search(r"The score is ([01])(?!\d)", text, re.IGNORECASE)
  if m:
    return float(m.group(1))
  m = re.search(r"([01])\s*$", text.strip())
  if m:
    return float(m.group(1))
  m = re.search(r"\b([01])\b", text)
  return float(m.group(1)) if m else 0.0

def judge_correctness(
  title: str,
  question: str,
  gt_evidence: list[str],
  gt_answer: list[str],
  prediction: str,
  times: int = 3,
) -> float:
  """Run judge `times` times and return majority-vote score (0.0 or 1.0)."""
  if not prediction.strip() or prediction in {"N/A", "n/a", "None", "Answer:"}:
    return 0.0
  if _IDK_RE.search(prediction):
    return 0.0
  query = {
    "document":   f"Paper title: {title}\n" + "\n".join(gt_evidence),
    "question":   question,
    "answer":     " ".join(gt_answer),
    "prediction": prediction,
  }
  scores: list[float] = []
  for _ in range(times):
    try:
      raw = _run_judge(query)
      scores.append(_extract_score(raw))
    except Exception as exc:
      print(f"  [judge error: {exc}]", file=sys.stderr)
  if not scores:
    return 0.0
  return 1.0 if sum(scores) / len(scores) >= 0.5 else 0.0

def _resolve_path(path: str) -> Path:
  p = Path(path)
  if p.is_absolute() or p.exists():
    return p
  script_relative = SCRIPT_DIR / p
  return script_relative if script_relative.exists() else p

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
dataset_path = _resolve_path(args.dataset)
results_path = _resolve_path(args.results)

try:
  with open(dataset_path, encoding="utf-8") as f:
    gt_map = {item["title"]: item for item in json.load(f)}
except FileNotFoundError:
  sys.exit(f"Dataset not found: {dataset_path}")

try:
  with open(results_path, encoding="utf-8") as f:
    results = json.load(f)
except FileNotFoundError:
  sys.exit(f"Results file not found: {results_path}")
except json.JSONDecodeError as e:
  sys.exit(f"Failed to parse {results_path}: {e}")

if not isinstance(results, list):
  sys.exit("Results JSON must be a list of objects.")

print(f"Judge API: provider={LLM_PROVIDER}  model={LLM_MODEL}  base_url={LLM_BASE_URL}")
print(f"Scoring {len(results)} entries against {len(gt_map)} papers in public dataset.\n")

# ──────────────────────────────────────────────────────────────────────────────
# Score
# ──────────────────────────────────────────────────────────────────────────────
per_paper: list[dict] = []
evid_total = corr_total = 0.0

for idx, item in enumerate(results, 1):
  title    = item.get("title", "")
  answer   = item.get("answer", "")
  evidence = item.get("evidence", [])

  gt = gt_map.get(title)
  if gt is None:
    print(f"[{idx:>3}] ⚠ title not found in public dataset — skipping")
    continue

  evid_score = calc_evidence_score(evidence, gt["evidence"])
  corr_score = judge_correctness(title, gt["question"], gt["evidence"], gt["answer"], answer, args.times)

  evid_total += evid_score
  corr_total += corr_score
  n = len(per_paper) + 1

  print(
    f"[{idx:>3}] evid={evid_score:.4f}  corr={corr_score:.0f}"
    f"   (running avg  evid={evid_total/n:.4f}  corr={corr_total/n:.4f})"
  )

  per_paper.append({
    "title":          title,
    "evidence_score": evid_score,
    "correctness":    corr_score,
    "answer":         answer,
  })

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
n = len(per_paper)
if n == 0:
  sys.exit("No papers scored.")

avg_evid = evid_total / n
avg_corr = corr_total / n

print(f"\n{'─'*60}")
print(f"  Papers scored  : {n} / {len(results)}")
print(f"  Evidence Score : {avg_evid:.5f}   (ROUGE-L, same as TA)")
print(f"  Correctness    : {avg_corr:.5f}   (LLM judge, approximate)")
print(f"{'─'*60}")
print(
  "  ⚠  Correctness is an approximation.\n"
  "     TA grading uses a separate model with a different evaluation procedure.\n"
  "     Use this score as a development signal, not as a prediction of your final grade."
)

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────
out_path = str(results_path.with_suffix("")) + "_score.json"
with open(out_path, "w", encoding="utf-8") as f:
  json.dump({
    "summary": {
      "n_scored":       n,
      "n_total":        len(results),
      "evidence_score": round(avg_evid, 6),
      "correctness":    round(avg_corr, 6),
    },
    "per_paper": per_paper,
  }, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {out_path}")
