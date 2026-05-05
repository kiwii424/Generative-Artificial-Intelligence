import argparse
import json
import os
import random
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm
import faiss
import nltk
from datasets import Dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from parse import parse_pdf, build_chunks, Chunk

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
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LR = 5e-5
NUM_EPOCHS = 2
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 0.5
EVAL_STEPS = 30
SAVE_STEPS = 30
EFF_NUM_BETA = 0.999

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


def parse_and_retrieve(split_df, pdf_dir, parsed_cache_dir, retrieved_cache_dir,
                       embed_model, reranker):
    Path(parsed_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(retrieved_cache_dir).mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in split_df.itertuples():
        grouped[row.paper_id].append({"id": row.id, "text": row.text,
                                       "label": getattr(row, "label", None)})
    for paper_id, samples in tqdm(grouped.items(), desc="parse+retrieve"):
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
            results[s["id"]] = {"text": s["text"], "label": s["label"], "evidence": evidence}
        with open(out_path, "w") as f:
            json.dump(results, f)
        del retr
        torch.cuda.empty_cache()


def load_retrieved_cache(retrieved_cache_dir, paper_id, sample_id):
    with open(Path(retrieved_cache_dir) / f"{paper_id}.json") as f:
        results = json.load(f)
    return results[str(sample_id)]


def get_evidence_for_row(retrieved_cache_dir, paper_id, sample_id, review_text=""):
    try:
        rec = load_retrieved_cache(retrieved_cache_dir, paper_id, sample_id)
        return format_evidence(rec["evidence"], review_text=review_text)
    except (FileNotFoundError, KeyError):
        cache_path = Path(retrieved_cache_dir) / f"{paper_id}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                results = json.load(f)
            if results:
                first_key = list(results.keys())[0]
                return format_evidence(results[first_key]["evidence"], review_text=review_text)
        return "[no evidence available]"


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


def build_prompt_msgs(system_prompt, user_template, review, evidence_text):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(evidence=evidence_text, review=review)},
    ]


class HW3Trainer(SFTTrainer):
    def __init__(self, *args, train_label_ids=None, use_sampler=False,
                 class_weights=None, n_classes=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_label_ids = train_label_ids
        self.use_sampler = use_sampler
        self.class_weights_np = class_weights
        self.n_classes = n_classes

    def _get_train_sampler(self, *args, **kwargs):
        if self.use_sampler and self.train_label_ids is not None:
            label_arr = np.array(self.train_label_ids)
            cw = self.class_weights_np if self.class_weights_np is not None else np.ones(self.n_classes)
            sample_weights = cw[label_arr]
            return WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(),
                                         num_samples=len(label_arr), replacement=True)
        return super()._get_train_sampler(*args, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--cache_dir", default="data_cache")
    ap.add_argument("--adapter_dir", default="adapter_checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for training.")
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    BF16 = torch.cuda.is_bf16_supported()
    FP16 = not BF16

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    parsed_cache_dir = cache_dir / "parsed_pdfs"
    retrieved_train_dir = cache_dir / "retrived" / "train-2"
    retrieved_dev_dir = cache_dir / "retrived" / "dev-2"

    print("Loading dataset")
    classes = json.load(open(data_dir / "classes.json"))
    train_df = pd.read_csv(data_dir / "train.csv")
    dev_df = pd.read_csv(data_dir / "dev.csv")
    id2concept = {c["id"]: c["concept"] for c in classes}
    class_defs = "\n".join(f"- {c['concept']}: {c['concept_desc']}" for c in classes)

    print("Loading embed + reranker models")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
    reranker = CrossEncoder(RERANKER_NAME, max_length=1024, device="cuda")

    print("Parse + retrieve train PDFs")
    parse_and_retrieve(train_df, data_dir / "paper_evidence" / "train",
                       parsed_cache_dir, retrieved_train_dir, embed_model, reranker)
    print("Parse + retrieve dev PDFs")
    parse_and_retrieve(dev_df, data_dir / "paper_evidence" / "dev",
                       parsed_cache_dir, retrieved_dev_dir, embed_model, reranker)

    del embed_model, reranker
    torch.cuda.empty_cache()

    print("Loading Qwen2.5-3B + LoRA")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=DTYPE, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES, lora_dropout=LORA_DROPOUT,
        bias="none", use_gradient_checkpointing="unsloth", random_state=SEED)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    SYSTEM_PROMPT, USER_TEMPLATE = make_prompt(class_defs)

    print("Building training texts")
    train_texts, train_label_ids = [], []
    for row in tqdm(train_df.to_dict("records"), desc="build train"):
        ev_text = get_evidence_for_row(retrieved_train_dir, row["paper_id"], row["id"],
                                       review_text=row["text"])
        msgs = build_prompt_msgs(SYSTEM_PROMPT, USER_TEMPLATE, row["text"], ev_text)
        msgs.append({"role": "assistant", "content": id2concept[int(row["label"])]})
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        train_texts.append(full)
        train_label_ids.append(int(row["label"]))

    dev_eval_df = dev_df.sample(min(300, len(dev_df)), random_state=SEED)
    dev_texts = []
    for row in tqdm(dev_eval_df.to_dict("records"), desc="build dev"):
        ev_text = get_evidence_for_row(retrieved_dev_dir, row["paper_id"], row["id"],
                                       review_text=row["text"])
        msgs = build_prompt_msgs(SYSTEM_PROMPT, USER_TEMPLATE, row["text"], ev_text)
        msgs.append({"role": "assistant", "content": id2concept[int(row["label"])]})
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        dev_texts.append(full)

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": dev_texts})

    label_counts = train_df["label"].value_counts().sort_index().values
    effective_num = (1.0 - np.power(EFF_NUM_BETA, label_counts)) / (1.0 - EFF_NUM_BETA)
    class_weights = (1.0 / effective_num)
    class_weights = class_weights / class_weights.mean()

    _dummy = [{"role": "user", "content": "x"}]
    _user_only_ids = tokenizer.apply_chat_template(_dummy, tokenize=True, add_generation_prompt=True)
    response_template_ids = _user_only_ids[-3:]
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)

    output_dir = Path(args.adapter_dir).parent / f"_train_run_seed{SEED}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir), max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text", packing=False,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO, num_train_epochs=NUM_EPOCHS,
        learning_rate=LR, max_grad_norm=MAX_GRAD_NORM,
        fp16=FP16, bf16=BF16, logging_steps=10,
        optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine",
        seed=SEED, eval_strategy="steps", eval_steps=EVAL_STEPS,
        save_strategy="steps", save_steps=SAVE_STEPS, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to="none",
    )

    trainer = HW3Trainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        args=training_args, data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        train_label_ids=train_label_ids, use_sampler=True,
        class_weights=class_weights, n_classes=len(classes),
    )

    print("Training")
    trainer.train()

    final_dir = Path(args.adapter_dir) / f"seed_{SEED}"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved adapter to {final_dir}")


if __name__ == "__main__":
    main()
