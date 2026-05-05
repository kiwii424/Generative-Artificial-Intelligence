# HW3: Retrieval-Augmented Hallucination Type Classification

This project builds a reproducible LLM pipeline for classifying hallucination types in academic peer reviews.
Given a review sentence and the reviewed paper PDF, the system retrieves supporting evidence from the paper and predicts one of five hallucination categories:

`Attribution Failure`, `Entity`, `Number`, `Overgeneralization`, `Temporal`

The final system combines **PDF parsing, hybrid retrieval, cross-encoder reranking, long-context QLoRA fine-tuning, class-imbalance sampling, and 3-seed deterministic ensemble voting**. The best public leaderboard score reached approximately **0.778**.

## At A Glance


| Area            | Implementation                                             |
| --------------- | ---------------------------------------------------------- |
| Task            | Hallucination type classification for peer review claims   |
| Evidence source | Full paper PDFs                                            |
| Retrieval       | BGE dense retrieval + BM25 + RRF + cross-encoder reranking |
| Model           | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`                     |
| Fine-tuning     | 4-bit QLoRA, LoRA`r=64`, `alpha=128`                       |
| Context length  | `MAX_SEQ_LENGTH = 10000`                                   |
| Prompt          | `cot_lite v3`, label-only output                           |
| Inference       | Greedy decoding, fixed 3-seed majority vote                |
| Score           | Public leaderboard around**0.778** (**Macro F1**)         |

## Why This Task Is Hard

This is not a simple text classification task. The model must compare a reviewer’s claim against the actual paper, and the boundary between classes is subtle:

- `Attribution Failure`: the review misstates what a cited source, prior work, or paper section says.
- `Entity`: the review inserts, swaps, or misnames a concrete object such as a model, method, dataset, metric, paper, or organization.
- `Number`: the exact value, percentage, count, rank, or measurement is wrong.
- `Overgeneralization`: a local result is stretched into a universal or future-general claim.
- `Temporal`: the error is about time, date, tense, or modality.

Many failures are close boundary cases: for example, a sentence can mention both a citation and a model name, or contain a number that is not actually the wrong part of the claim. The pipeline is designed around these boundary decisions.

## System Architecture

```text
Paper PDFs
   |
   v
PDF parser + section detector
   |
   v
Section-aware chunking
   |
   v
Hybrid retrieval
   |-- BGE dense search
   |-- BM25 lexical search
   |-- Reciprocal Rank Fusion
   v
Cross-encoder reranker
   |
   v
Evidence formatter
   |
   v
cot_lite v3 prompt
   |
   v
Qwen2.5-3B QLoRA classifier
   |
   v
Greedy prediction per seed
   |
   v
3-seed majority vote
   |
   v
Submission CSV
```

## Key Design Choices

### 1. Retrieval Before Classification

The review sentence alone is often insufficient. The system first retrieves evidence from the paper so the model can judge whether the claim is supported, contradicted, or unsupported.

The retrieval stack intentionally combines semantic and lexical methods:


| Component              | Why it matters                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------- |
| BGE dense retrieval    | Finds semantically related paper passages                                             |
| BM25                   | Catches exact anchors such as table names, citations, metrics, years, and model names |
| RRF                    | Stabilizes ranking by fusing dense and lexical signals                                |
| Cross-encoder reranker | Reorders candidates using direct query-passage relevance                              |

This matters because hallucination labels often depend on exact anchors, not only semantic similarity.

### 2. Section-Aware PDF Parsing

[parse.py](./parse.py) reconstructs paper structure using:

- font-size and boldness signals
- numbered and lettered heading patterns
- keyword-based section mapping
- filtering for administrative sections when they are not relevant

The parser maps paper text into sections such as `abstract`, `introduction`, `methodology`, `experiments`, `results`, and `conclusion`, then creates long but bounded chunks.

Final retrieval chunk settings:


| Parameter            | Value |
| -------------------- | ----- |
| `CHUNK_TARGET_CHARS` | 6800  |
| `CHUNK_MAX_CHARS`    | 9000  |
| `MERGE_MIN_CHARS`    | 3000  |
| `EVIDENCE_TOP_K`     | 5     |
| `MAX_EVIDENCE_CHARS` | 10000 |

This setting preserves enough context for paper-level claims while keeping prompts within the model context limit.

### 3. Boundary-Aware Prompting

The final prompt, `cot_lite v3`, gives the model compact internal decision rules while forcing a single category output. It avoids verbose reasoning in the generated answer and keeps the task aligned with the submission format.

The prompt’s main idea is:

```text
Choose the class by identifying which part of the review claim is wrong:
- cited source content -> Attribution Failure
- named object -> Entity
- broadness / universality -> Overgeneralization
- exact numeric value -> Number
- timing / tense / modality -> Temporal
```

This was especially useful for reducing false triggers where any number became `Number`, or any named method became `Entity`.

### 4. Long-Context QLoRA

[train.py](./train.py) fine-tunes `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` with QLoRA:


| Parameter                     | Value        |
| ----------------------------- | ------------ |
| `MAX_SEQ_LENGTH`              | 10000        |
| `LORA_R`                      | 64           |
| `LORA_ALPHA`                  | 128          |
| `LORA_DROPOUT`                | 0.05         |
| `learning_rate`               | 5e-5         |
| `num_train_epochs`            | 2            |
| `per_device_train_batch_size` | 1            |
| `gradient_accumulation_steps` | 16           |
| `warmup_ratio`                | 0.05         |
| optimizer                     | `adamw_8bit` |
| scheduler                     | cosine       |

Training uses `DataCollatorForCompletionOnlyLM`, so the loss is applied only to the assistant answer span. This keeps the model focused on producing the class label rather than reconstructing the prompt or retrieved evidence.

### 5. Class Imbalance Handling

The dataset is dominated by the three large classes, while `Number` and `Temporal` are much smaller. The training script uses effective-number class weights with `WeightedRandomSampler` to give minority classes more exposure without changing the task format.

### 6. Deterministic 3-Seed Ensemble

[inference.py](./inference.py) runs greedy decoding for each trained seed:

```text
temperature = 0.0
do_sample = False
max_new_tokens = 12
```

The final prediction is a fixed-order majority vote:

```text
seed_42
seed_9222
seed_786349
```

This improves stability on ambiguous boundary samples while keeping inference deterministic and reproducible.

## Results And Error Analysis

Final submitted configuration:

```text
Retrieval: R0
Chunking: target=6800, max=9000, merge_min=3000
Top-k evidence: 5
Max evidence chars: 10000
Prompt: cot_lite v3
Model: Qwen2.5-3B-Instruct 4-bit + LoRA r=64 alpha=128
Inference: fixed 3-seed greedy ensemble
Postprocess: deterministic label parsing + majority vote
```

The best public leaderboard score reached approximately **0.778**.

The strongest remaining error modes are boundary confusions among:

- `Entity` vs `Attribution Failure`
- `Overgeneralization` vs `Entity`
- `Attribution Failure` vs `Overgeneralization`

This matches the nature of the task: many review sentences contain both a named object and a broader unsupported claim, so the classifier must decide which part is actually hallucinated.

## How To Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The pipeline expects a CUDA GPU. The final configuration was designed for an A100-class environment because long-context QLoRA with `MAX_SEQ_LENGTH=10000` is memory-intensive.

### 2. Prepare Dataset

Expected layout:

```text
dataset/
├── classes.json
├── train.csv
├── dev.csv
├── test.csv
└── paper_evidence/
    ├── train/
    ├── dev/
    └── test/
```

### 3. Run Inference

If trained adapters are available under `adapter_checkpoint/`, run:

```bash
python inference.py \
    --data_dir dataset \
    --cache_dir data_cache \
    --adapter_dir adapter_checkpoint \
    --output_csv hw3_111511157.csv
```

The script will:

1. Parse test PDFs
2. Build retrieval cache
3. Load seed adapters in fixed order
4. Generate one greedy prediction per seed
5. Majority-vote labels
6. Write `hw3_111511157.csv`

### 4. Retrain From Scratch

```bash
python train.py --seed 42
python train.py --seed 9222
python train.py --seed 786349
```

Each command saves one adapter under:

```text
adapter_checkpoint/seed_<SEED>/
```

After all three seeds are trained, run `inference.py` to generate the final ensemble submission.

## Repository Structure

```text
HW3/
├── main.ipynb                   # Experiment colab notebook
├── README.md                    # Project overview and reproduction guide
├── explanation.md               # Step-by-step notebook explanation
├── parse.py                     # PDF parser and chunker
├── train.py                     # Retrieval + QLoRA training
├── inference.py                 # Retrieval + deterministic ensemble inference
├── requirements.txt
└── main.zip                     # Final package including adapter_checkpoint
```

## What This Project Demonstrates

- Building a retrieval-augmented classifier from raw research PDFs
- Designing hybrid retrieval for both semantic and exact-match evidence
- Engineering a PDF parser robust enough for noisy academic layouts
- Using long-context QLoRA to adapt a 3B instruction model
- Applying completion-only loss for label-generation classification
- Handling severe class imbalance with sampler-based training
- Using deterministic multi-seed ensembling for leaderboard stability
- Performing error analysis on subtle hallucination taxonomy boundaries

## Limitations And Next Steps

- The hardest cases are semantic boundary cases rather than obvious formatting errors.
- Retrieval quality directly controls classifier quality; missing evidence can change the predicted class.
- Long context improves evidence coverage but increases VRAM usage.
- Future work could add citation-aware retrieval, calibrated ensemble voting, or class-specific evidence formatting.
