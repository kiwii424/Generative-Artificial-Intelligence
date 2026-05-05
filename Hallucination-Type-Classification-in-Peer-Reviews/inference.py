import argparse
import csv
import gc
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import faiss
import nltk
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from HW3.parse import parse_pdf, build_chunks

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


CHUNK_TARGET_CHARS = 6800
CHUNK_MAX_CHARS = 9000
MERGE_MIN_CHARS = 3000
EVIDENCE_TOP_K = 5
MAX_EVIDENCE_CHARS = 10000
DENSE_TOP_K = 60
BM25_TOP_K = 60
RRF_K = 60
RRF_TOP_K = 30
RERANK_POOL = 30

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_BATCH = 32
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

MAX_SEQ_LENGTH = 10000
GEN_MAX_NEW_TOKENS = 12
ENSEMBLE_SEEDS = [42, 9222, 786349]

REFERENCE_QUERY_PAT = re.compile(
    r"\b(references?|citation|citations|cite|cited|citing|referenced|bibliograph|uncited)\b", re.I)
CHECKLIST_QUERY_PAT = re.compile(
    r"\b(checklist|irb|ethics?|code of ethics|broader impacts?|institutional review board)\b", re.I)


def _looks_like_reference_evidence(ev):
    text = str(ev.get("text", ""))
    head = text[:1400].lower()
    if re.match(r"^\s*(references|bibliography)\b", head):
        return True
    if re.match(r"^\s*\[\d+\]\s+", text):
        return True
    bracket_refs = len(re.findall(r"\[\d+\]", head))
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", head))
    ref_terms = sum(t in head for t in [
        "arxiv", "doi", "proceedings", "journal", "conference",
        "pmlr", "neurips", "icml", "iclr", "cvpr", "url http"])
    return (bracket_refs >= 3) or (years >= 4 and ref_terms >= 1)


def _looks_like_checklist_evidence(ev):
    text = str(ev.get("text", ""))
    low = text[:2500].lower()
    if "neurips paper checklist" in low or "paper checklist" in low:
        return True
    terms = ["question: does the paper", "answer: [", "justification:", "guidelines:",
             "experimental result reproducibility", "experiments compute resources",
             "open access to data and code", "code of ethics question",
             "broader impacts question", "institutional review board"]
    return sum(t in low for t in terms) >= 3


def _filter_admin_evidence(evidence, review_text=""):
    allow_refs = bool(REFERENCE_QUERY_PAT.search(review_text or ""))
    allow_checklist = bool(CHECKLIST_QUERY_PAT.search(review_text or ""))
    filtered = []
    for ev in evidence:
        if not allow_checklist and _looks_like_checklist_evidence(ev):
            continue
        if not allow_refs and _looks_like_reference_evidence(ev):
            continue
        filtered.append(ev)
    return filtered if filtered else evidence[:1]


def format_evidence(evidence, max_chars=MAX_EVIDENCE_CHARS, review_text=""):
    ev_filtered = _filter_admin_evidence(evidence, review_text=review_text)
    ev_sorted = sorted(ev_filtered, key=lambda x: x.get("score", 0), reverse=True)
    parts, total = [], 0
    for ev in ev_sorted:
        chunk = f"[{ev['section']}] {ev['text']}"
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            if remaining >= 800:
                parts.append(chunk[:remaining].rstrip())
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(parts)


def _tok(text):
    return re.findall(r"\w+", text.lower())


class PaperRetriever:
    def __init__(self, chunks, embed_model, reranker):
        self.chunks = chunks
        self.embed_model = embed_model
        self.reranker = reranker
        if not chunks:
            self.index = self.bm25 = None
            return
        texts = [c.text for c in chunks]
        embs = embed_model.encode(texts, batch_size=EMBED_BATCH,
                                  normalize_embeddings=True, show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.bm25 = BM25Okapi([_tok(t) for t in texts])

    def dense(self, q, top_k):
        if self.index is None:
            return []
        emb = self.embed_model.encode([EMBED_QUERY_PREFIX + q],
                                      normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(emb, min(top_k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def bm25_search(self, q, top_k):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tok(q))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(i, float(scores[i])) for i in ranked]

    @staticmethod
    def rrf_fuse(*rls, k=RRF_K):
        sc = defaultdict(float)
        for rl in rls:
            for rank, (cid, _) in enumerate(rl):
                sc[cid] += 1.0 / (k + rank + 1)
        return sorted(sc, key=lambda x: -sc[x])

    def rerank(self, q, ids, top_k):
        if not ids:
            return []
        pairs = [(q, self.chunks[i].text) for i in ids]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        return sorted(zip(ids, [float(s) for s in scores]), key=lambda x: -x[1])[:top_k]

    def retrieve(self, query, top_k=EVIDENCE_TOP_K):
        dense_hits = self.dense(query, DENSE_TOP_K)
        bm25_hits = self.bm25_search(query, BM25_TOP_K)
        fused = self.rrf_fuse(dense_hits, bm25_hits)[:RRF_TOP_K]
        reranked = self.rerank(query, fused, top_k=RERANK_POOL)
        out = reranked[:top_k]
        return [{"chunk_id": cid, "score": s, "text": self.chunks[cid].text,
                 "section": self.chunks[cid].section} for cid, s in out]


def make_prompt(class_defs):
    SYSTEM_PROMPT = (
        "You are an expert reviewer evaluating academic peer review sentences for hallucinations.\n\n"
        "Compare the review sentence against the paper context. Internally answer these in order, then choose ONE label:\n\n"
        "Q1. Does the review explicitly cite or refer to a specific source ([N], \"the paper by X\", \"in [Smith 2020]\") AND make a claim about what that source says, proves, or does?\n"
        "   -> If YES and the cited source's content does not actually say what the review claims -> Attribution Failure\n"
        "   -> This rule wins over Entity when the review is about a citation's content, not its existence.\n\n"
        "Q2. Does the review name a specific entity (model, dataset, method, metric, person, paper title, agent) that does not appear in the evidence, or appears with a different name?\n"
        "   -> If YES and Q1 did not fire -> Entity\n\n"
        "Q3. Does the review extend a specific result/finding into a universal, all-cases, always, fully-proven, or future-general claim?\n"
        "   -> If YES -> Overgeneralization\n\n"
        "Q4. Does the review state a SPECIFIC numeric value AND that exact value is wrong vs the evidence?\n"
        "   -> \"achieves 92% accuracy\" but evidence shows 85% -> Number\n"
        "   -> Just mentioning a number (\"speedup of 2.2x\", \"with 98.7% accuracy\") without the value being challenged -> NOT Number; recheck Q1/Q2/Q3\n"
        "   -> Words like \"numerically\" or \"second place\" alone -> NOT Number\n\n"
        "Q5. Is the wrongness about TIME, DATE, TENSE, or MODALITY?\n"
        "   -> Year/date that disagrees with evidence (e.g., \"in 2009\" when paper is 2018) -> Temporal\n"
        "   -> Modal verbs implying wrong certainty about future/past (\"must perform\", \"should outperform\", \"will achieve\", \"had been\") -> Temporal\n"
        "   -> NOT just any past-tense verb; only when the timing/modality is the WRONG part\n\n"
        f"Categories:\n{class_defs}\n\n"
        "Tie-breaker (when two rules fire, choose by WHAT PART is the wrong content):\n"
        "- timing / date / tense / modality wrong -> Temporal\n"
        "- exact numeric value wrong -> Number\n"
        "- named object wrong / missing / swapped -> Entity\n"
        "- claim about a cited source's content unsupported -> Attribution Failure\n"
        "- local result extended too broadly -> Overgeneralization\n\n"
        "Output ONLY the category name exactly as written. No explanation. No quotes."
    )
    USER_TEMPLATE = "Paper Context:\n{evidence}\n\nReview Sentence:\n{review}\n\nHallucination Type:"
    return SYSTEM_PROMPT, USER_TEMPLATE


def parse_label(output, classes):
    out = output.strip().lower()
    for cls in classes:
        if cls["concept"].lower() in out:
            return cls["id"]
    if "attribut" in out: return 0
    if "entity" in out: return 1
    if "number" in out: return 2
    if "overgeneral" in out or "general" in out: return 3
    if "temporal" in out or "tense" in out: return 4
    return 0


@torch.no_grad()
def predict_one(model, tokenizer, system_prompt, user_template, review, evidence_text,
                classes, temp=0.0):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(evidence=evidence_text, review=review)},
    ]
    prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    do_sample = temp > 0.0
    out = model.generate(
        **inputs, max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=do_sample,
        temperature=(temp if do_sample else 1.0),
        top_p=(0.9 if do_sample else 1.0),
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_label(text, classes), text


def parse_and_retrieve_test(test_df, pdf_dir, parsed_cache_dir, retrieved_cache_dir,
                            embed_model, reranker):
    Path(parsed_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(retrieved_cache_dir).mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in test_df.itertuples():
        grouped[row.paper_id].append({"id": row.id, "text": row.text})
    for paper_id, samples in tqdm(grouped.items(), desc="parse+retrieve test"):
        out_path = Path(retrieved_cache_dir) / f"{paper_id}.json"
        if out_path.exists():
            continue
        parsed_path = Path(parsed_cache_dir) / f"{paper_id}.json"
        if parsed_path.exists():
            with open(parsed_path) as f:
                parsed = json.load(f)
        else:
            pdf_path = Path(pdf_dir) / f"{paper_id}.pdf"
            if not pdf_path.exists():
                print(f"  warn: missing pdf {pdf_path}")
                continue
            parsed = parse_pdf(str(pdf_path))
            with open(parsed_path, "w") as f:
                json.dump(parsed, f)
        chunks = build_chunks(parsed, target_chars=CHUNK_TARGET_CHARS,
                              max_chars=CHUNK_MAX_CHARS, merge_min=MERGE_MIN_CHARS)
        if not chunks:
            continue
        retr = PaperRetriever(chunks, embed_model, reranker)
        results = {}
        for s in samples:
            evidence = retr.retrieve(s["text"], top_k=EVIDENCE_TOP_K)
            results[s["id"]] = {"text": s["text"], "evidence": evidence}
        with open(out_path, "w") as f:
            json.dump(results, f)
        del retr
        torch.cuda.empty_cache()


def load_evidence(retrieved_cache_dir, paper_id, sample_id, review_text):
    path = Path(retrieved_cache_dir) / f"{paper_id}.json"
    if not path.exists():
        return "[no evidence available]"
    with open(path) as f:
        results = json.load(f)
    rec = results.get(str(sample_id))
    if rec is None:
        first_key = list(results.keys())[0] if results else None
        if first_key is None:
            return "[no evidence available]"
        rec = results[first_key]
    return format_evidence(rec["evidence"], review_text=review_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--cache_dir", default="data_cache")
    ap.add_argument("--adapter_dir", default="adapter_checkpoint")
    ap.add_argument("--output_csv", default="hw3_111511157.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for inference.")
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    parsed_cache_dir = cache_dir / "parsed_pdfs"
    retrieved_test_dir = cache_dir / "retrived" / "test-2"

    print("Loading test set + classes")
    classes = json.load(open(data_dir / "classes.json"))
    test_df = pd.read_csv(data_dir / "test.csv")
    class_defs = "\n".join(f"- {c['concept']}: {c['concept_desc']}" for c in classes)

    print("Loading embed + reranker")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
    reranker = CrossEncoder(RERANKER_NAME, max_length=1024, device="cuda")

    print("Parse + retrieve test PDFs")
    parse_and_retrieve_test(test_df, data_dir / "paper_evidence" / "test",
                            parsed_cache_dir, retrieved_test_dir, embed_model, reranker)

    del embed_model, reranker
    torch.cuda.empty_cache()

    adapter_root = Path(args.adapter_dir)
    ordered_seed_adapters = [
        adapter_root / f"seed_{seed}"
        for seed in ENSEMBLE_SEEDS
        if (adapter_root / f"seed_{seed}" / "adapter_config.json").exists()
    ]
    sub_adapters = sorted(p for p in adapter_root.iterdir()
                          if p.is_dir() and (p / "adapter_config.json").exists())
    if ordered_seed_adapters:
        adapter_paths = ordered_seed_adapters
        print(f"Found {len(adapter_paths)} ordered seed adapters in {adapter_root}: {[p.name for p in adapter_paths]}")
    elif sub_adapters:
        adapter_paths = sub_adapters
        print(f"Found {len(adapter_paths)} seed adapters in {adapter_root}: {[p.name for p in adapter_paths]}")
    else:
        adapter_paths = [adapter_root]
        print(f"Using single adapter: {adapter_root}")

    SYSTEM_PROMPT, USER_TEMPLATE = make_prompt(class_defs)

    pre_loaded = []
    for row in test_df.to_dict("records"):
        ev_text = load_evidence(retrieved_test_dir, row["paper_id"], row["id"], row["text"])
        pre_loaded.append((int(row["id"]), row["text"], ev_text))

    from unsloth import FastLanguageModel
    all_seed_preds = []
    for ai, adapter_path in enumerate(adapter_paths):
        print(f"\n[adapter {ai+1}/{len(adapter_paths)}] loading {adapter_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path), max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE, load_in_4bit=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "right"
        FastLanguageModel.for_inference(model)

        seed_preds = {}
        for sid, review, ev_text in tqdm(pre_loaded, desc=f"predict ({adapter_path.name})"):
            pid, _ = predict_one(model, tokenizer, SYSTEM_PROMPT, USER_TEMPLATE,
                                 review, ev_text, classes, temp=0.0)
            seed_preds[sid] = int(pid)
        all_seed_preds.append(seed_preds)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for sid, _, _ in pre_loaded:
        votes = [sp.get(sid) for sp in all_seed_preds if sp.get(sid) is not None]
        if not votes:
            winner = 0
        else:
            winner = Counter(votes).most_common(1)[0][0]
        rows.append({"id": sid, "label": int(winner)})

    out_path = Path(args.output_csv)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for r in rows:
            w.writerow([r["id"], r["label"]])
    print(f"Wrote {len(rows)} predictions ({len(adapter_paths)}-seed ensemble) to {out_path}")


if __name__ == "__main__":
    main()
