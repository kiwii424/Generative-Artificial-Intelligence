# HW2: Document QA based on RAG
> 535109 生成式人工智慧   ·  2026 Spring Semester
> Author: Wei-Chih, Chen (陳偉銍)
> 2026/03/31

---

## Introduction

本次作業將引導大家完成一個文件知識問答系統，我們將以 200 篇 NLP 領域的論文作為實驗對象。

請使用 RAG (Retrieval-Augmented Generation) 技術，基於提供的資料集實作 Document QA。

- 本次 **LLM 統一使用 TA 指定的模型**，不允許自行更換
- Embedding model 與 Rerank model 可自行選擇**免費且開源**的模型
- 需要設計：
  - Query LLM 的 Prompt
  - 一套回答問題的 Pipeline（Pipeline 中可以進行多輪/多階段 LLM 對話，例如對搜尋到的文字進行摘要等）

---

## 指定 LLM

**本次統一使用：`meta-llama/Llama-3.2-3B-Instruct`**

> 固定 LLM 的目的是讓大家將重點放在 RAG Pipeline 本身的設計（Embedding、Chunking、Reranking、Prompting 策略），而非靠更大的模型取勝。

存取方式（任選其一即可）：
- **Ollama（本機）**：`ollama run llama3.2:3b`，並透過 [langchain-ollama](https://python.langchain.com/docs/integrations/llms/ollama/) 或 OpenAI 相容 API 呼叫
- **OpenRouter API（雲端免費）**：使用 `meta-llama/llama-3.2-3b-instruct:free`
- **vLLM / HuggingFace TGI（自架）**：如有 GPU 資源可自行啟動 OpenAI 相容 server

**不允許使用**：其他 LLM（包含更大的模型、其他開源或商業模型）

---

## Embedding / Rerank Model 規範

可自行選擇**任何免費且開源**的 embedding model 或 rerank model，但須符合以下條件：

- ✅ 可使用：HuggingFace 上的開源模型（例如 `BAAI/bge-*`、`mixedbread-ai/mxbai-*`、`Snowflake/snowflake-arctic-embed-*`、`intfloat/e5-*` 等）
- ✅ 可透過 Ollama、sentence-transformers、infinity-emb 等本機方式載入
- ❌ **不可使用**：需要付費或使用雲端平台免費額度的 API（例如 OpenAI Embeddings、Google Vertex AI、Cohere Embed 等）
- 最終 submission 只能使用**一個** embedding model 

---

## Dataset & Files

- **public_dataset.json**：包含 100 篇論文及對應問答，供開發使用（含 answer 與 evidence）
- **private_dataset.json**：包含 100 篇論文及對應問題，供最終排名使用（**不含** answer 與 evidence）
- **sample_submission.json**：正確格式的範例 submission

### Public Dataset 格式預覽

```json
{
  "title": "Semi-supervised Thai Sentence Segmentation Using Local and Distant Word Representations",
  "full_text": "Abstract\nA sentence is typically...\n\n\nComparison of CNN and n-gram models for ...\n...\n\n\n",
  "question": "How do they utilize unlabeled data to improve model representations?",
  "answer": [
    "During training, the model is trained alternately with ... mini-batches of unlabeled data."
  ],
  "evidence": [
    "CVT BIBREF20 is a semi-supervised learning technique whose...ata.",
    "Labeled data are input into the model to calculate the sta...ess.",
    "As discussed in Section SECREF3 , CVT requires primary and...FORM0"
  ]
}
```

---

## Evaluation Metrics

針對 Document QA 系統，我們評估兩件事：

### 1. Answer Correctness（由 LLM 評分）
- 每題得 0 或 1 分

### 2. Evidence Score（ROUGE-L F-measure）
- 計算所有 retrieved documents 與 ground truth evidence 的 ROUGE-L，並取最大值
- 若有多筆 ground truth evidence，取每筆各自最大 ROUGE-L 的平均
- 具體來說：有 n 筆 ground truth evidence，與 K 個 retrieved chunks 計算 ROUGE-L，每筆 evidence 取 K 個分數中的最大值，最後對 **K** 個最大值求平均，得到最終 evidence score  

Ref: [Text Summarization: How To Calculate Rouge Score | by Eren Kızılırmak | Medium](https://medium.com/@eren9677/text-summarization-387836c9e178) 

---

## Grading Policy（100%：競賽成績）

本學期**取消報告**，成績 100% 由 Private Dataset 的競賽結果決定。

### Evidence Score Ranking（50 分）

| 條件                                | 得分       |
| --------------------------------- | -------- |
| 超過 Weak Baseline（> **0.2124**）    | 60%      |
| 超過 Strong Baseline（> **0.26185**） | 80%      |
| 排名 50%～75%                        | +10%     |
| 排名 20%～50%                        | +15%     |
| 排名前 20%                           | +20%（全拿） |

### Answer Correctness Ranking（50 分）

| 條件                             | 得分       |
| ------------------------------ | -------- |
| 超過 Weak Baseline（> **0.33**）   | 60%      |
| 超過 Strong Baseline（> **0.48**） | 80%      |
| 排名 50%～75%                     | +10%     |
| 排名 20%～50%                     | +15%     |
| 排名前 20%                        | +20%（全拿） |


---

## 探索方向參考（鼓勵嘗試，不列入評分）

以下為一些可能提升 RAG 效果的方向，歡迎大家深入探索：

### Chunking 策略
- 調整 chunk size 與 overlap 大小
- 使用語意分割（Semantic Chunking）取代固定字元切割
- 嘗試 Hierarchical / Parent-Child Chunking

### Retrieval 策略
- 調整每題 retrieved chunks 的數量（`evidence` list 長度須滿足 **1 ≤ k ≤ 40**，各題可不同）
- 混合搜尋（Hybrid Search）：向量搜尋 + BM25 關鍵字搜尋
- 使用 Rerank model 對 retrieved chunks 重新排序
- Multi-Query：對同一問題生成多個查詢語句以擴大搜尋範圍

### Prompt 技巧 (僅供參考)
- Chain-of-Thought (CoT)
- 指定輸出格式或長度限制
- 加入 context 的結構化整理指令

### 進階技術
- **HyDE（Hypothetical Document Embeddings）**：先讓 LLM 針對問題生成一段「假設答案」，再將這段假設答案的 embedding 作為查詢向量進行搜尋，通常比直接用問題搜尋更精準。
  > 論文：[Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) (Gao et al., 2022)
- **RAG-Fusion / Query Expansion**：讓 LLM 從不同角度生成多個查詢語句（Multi-Query），分別搜尋後合併結果，再用 Reciprocal Rank Fusion (RRF) 重新排序，能有效提升召回率。
  > 論文：[RAG-Fusion: a New Take on Retrieval-Augmented Generation](https://arxiv.org/abs/2402.03367) (Raudaschl, 2024)  
  > LangChain 實作：[MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)

請注意，**不可用答案回去找evidence**。不接受使用RAG回答的結果再回頭去retrieve，以提高evidence score。
**LLM 回答的依據必須是提交的evidence。**

---

## 套件管理

推薦使用 **[uv](https://docs.astral.sh/uv/)** 管理 Python 環境。

常用指令：
```bash
# 安裝 uv (https://docs.astral.sh/uv/getting-started/installation/#installation-methods)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 建立虛擬環境並安裝依賴
uv sync

# 執行程式
uv run python {student_id}.py
```

推薦套件（不限定，可依需求增加）：
```
langchain langchain-community langchain-ollama
sentence-transformers faiss-cpu
rich
rouge-score
```

---

## Important Notes

1. **嚴禁使用不公平手段**取得分數，違者競賽成績歸零（不要鋌而走險！）
2. **禁止 Fine-tuning**：不允許對任何特定資料集進行 fine-tune 的 LLM 或 embedding model
3. **禁止使用額外的付費或雲端 API 的 Embedding**（詳見 Embedding/Rerank Model 規範）
4. **不要將模型上傳至 E3**
5. Final submission 只能使用**一個** embedding model
6. 請嚴格遵守提交格式，**在上傳前務必於 Colab 或 Linux 系統上測試** `unzip hw2_{student_id}.zip`，確認解壓後的檔案結構正確
7. **每次 RAG 請以單篇論文（full_text）為單位處理**，不可跨論文建立向量庫
8. LLM 回答的依據必須是提交的evidence
9. LLM 必須使用 `meta-llama/Llama-3.2-3B-Instruct`，不可替換
10. 每題提交的 `evidence` list 長度須滿足 **1 ≤ k ≤ 40**，各題數量可不同（ground truth 的數量每題也不盡相同），**違規者扣2分。**

---

## Pre-submissions（前測）

| 次數  | 截止時間                    |
| --- | ----------------------- |
| 1st | 2026/04/06 (Mon.) 23:59 |
| 2nd | 2026/04/08 (Wed.) 23:59 |
| 3rd | 2026/04/13 (Mon.) 23:59 |
| 4th | 2026/04/15 (Wed.) 23:59 |
| 5th | 2026/04/19 (Sat.) 23:59 |
| 6th | 2026/04/20 (Mon.) 23:59 |

**提交方式**：請將 Private Dataset 的預測結果（100 筆）以 `{student_ID}.json` 格式上傳至 E3。

> 注意：格式錯誤將喪失前測機會。

---

## Final Submission

**截止時間**：2026/04/21 (Tue.) 23:59（繳交到E3）

**提交方式**：請將以下檔案壓縮為 `hw2_{student_id}.zip` 後上傳至 E3：

```
hw2_{student_id}/
├── {student_id}.py     ← 你的 RAG 程式碼
└── {student_id}.json   ← Private dataset 的 100 筆預測結果
```

結果必須可透過以下指令重現：
```bash
python {student_id}.py
```

**解壓範例**：
```bash
$ unzip hw2_111222333.zip
# 解壓後應出現：
hw2_111222333/
├── 111222333.py
└── 111222333.json
```

> **格式錯誤罰分**：-5 分  
> **遲交**：不影響其他同學排名。每天扣原始分數 5%，最晚補交至 2026/04/28 23:59（一週後）

---

## Try it out!

### 免費 GPU 環境推薦

| 平台 | GPU | 限制 | 備註 |
|------|-----|------|------|
| [Kaggle](https://www.kaggle.com/code) | P100 / T4 | 30 小時/週，單次最長 12 小時 | **推薦**，穩定好取得 |
| [Google Colab](https://colab.research.google.com) | T4（免費版越來越難取得） | 連線時間較短 | 建議搭配 Colab Pro |
| [Lightning AI](https://lightning.ai/) | T4 | 免費 22 小時/月 | 支援 VS Code 介面 |

> Llama-3.2-3B-instruct 僅需約 7-8 GB VRAM，P100 / T4 皆可執行 (vLLM更快但需要更多VRAM)

### 免費 LLM API（無需 GPU）

若不想自己跑模型，可使用以下平台的免費 API，**呼叫時須指定模型為 `meta-llama/llama-3.2-3b-instruct`**：

**OpenRouter（推薦）**：提供 `meta-llama/llama-3.2-3b-instruct:free`，無需付費。

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="meta-llama/llama-3.2-3b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    api_key="your_openrouter_api_key",  # https://openrouter.ai 申請免費帳號
    temperature=0.3,
    max_tokens=256,
)
```

> 其他免費替代平台（若 OpenRouter 有流量限制時）：
> - [Hugging Face Inference API](https://huggingface.co/inference-api)（免費額度）