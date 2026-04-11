# Generative Artificial Intelligence Projects

這個 repository 收錄我在生成式人工智慧課程中完成的兩個 NLP 專案，主題分別是醫學病理選擇題問答與以論文為基礎的文件問答系統。兩份作業都不是只停留在模型呼叫，而是從資料處理、prompt 設計、模型/檢索策略、評估到實驗迭代都完整實作。

如果你是招募方或工程主管，可以先看下面這張表：

| Project | Problem | What I built | Outcome |
| --- | --- | --- | --- |
| [HW1](./HW1) | Pathology multiple-choice QA | 以 `unsloth/Llama-3.2-1B-Instruct` 做 4-bit LoRA 微調，搭配 few-shot CoT prompt 與答案格式約束 | 內部分割驗證集準確率 **71.11%** |
| [HW2](./HW2) | Document QA over 200 NLP papers | 建立完整 RAG pipeline：parent-child chunking、FAISS + BM25 混合檢索、HyDE、RRF、reranker、LLM generation | 保存的公開測試版本顯示 evidence score 最高到 **0.2798**；另有系統化 ablation 實驗 |

## Why This Repo Is Worth Reading

- 把生成式 AI 真正落地成兩種不同問題型態：`instruction tuning` 與 `RAG`
- 不只使用模型，也處理了資料切分、prompt engineering、檢索設計、reranking、評估與錯誤分析
- 每份作業都有可追的主要實作檔，方便快速檢視我的思路與工程風格

## HW1: Pathology QA

目標是解決病理學單選題問答。資料格式包含題目、四個選項與正確答案，最後需要輸出符合競賽格式的提交結果。核心實作在 [HW1/main.ipynb](./HW1/main.ipynb)。

我採用的方法：

- 使用 `unsloth/Llama-3.2-1B-Instruct` 作為基底模型，透過 **4-bit quantization + LoRA** 降低訓練資源需求
- 設計 **few-shot prompt**，在 prompt 中加入病理學示例題，並要求模型只輸出 `A / B / C / D`
- 在 system prompt 中加入 **Chain-of-Thought 導向**，讓模型先做病理判斷再輸出最終答案，但在輸出格式上嚴格限制只保留選項字母
- 以 `SFTTrainer` 進行 supervised fine-tuning，並搭配 completion-only loss 讓模型專注學習答案區段
- 針對驗證流程實作自動化推論、regex 後處理、classification report 與 confusion matrix 分析

這份作業展現的能力：

- 小模型指令微調與 parameter-efficient fine-tuning
- Prompt 設計與輸出格式控制
- 醫學問答場景下的模型評估與誤差分析

目前 notebook 中保存的結果為：

- 內部分割驗證集準確率：**71.11%**
- Macro F1：約 **0.70**

## HW2: Document QA Based on RAG

目標是針對 200 篇 NLP 論文建立文件問答系統，必須根據每篇 paper 的 `full_text` 回答問題，並同時提交可支持答案的 evidence。核心實作在 [HW2/111511157.py](./HW2/111511157.py)，實驗工具在 [HW2/ablation_study.py](./HW2/ablation_study.py)。

我採用的方法：

- 使用 **sentence-based parent-child chunking**
  - child chunk 盡量短，提升 evidence 的 ROUGE-L 對齊能力
  - parent chunk 保留較完整上下文，提供 reranker 與 LLM 使用
- 建立 **hybrid retrieval**
  - dense retrieval：`BAAI/bge-large-en-v1.5` + FAISS
  - sparse retrieval：BM25
- 對同一個問題做 **query expansion**
  - 原始問題
  - 規則式 stripped query / keyword query
  - HyDE 生成 hypothetical passage 後再做向量檢索
- 以 **Reciprocal Rank Fusion (RRF)** 合併多路檢索結果，再用 `BAAI/bge-reranker-v2-m3` 做 cross-encoder reranking
- 依 reranker 分數分布做 **dynamic evidence selection**，控制每題提交的 evidence 數量
- 使用指定模型 `meta-llama/Llama-3.2-3B-Instruct` 產生最終答案，並用 prompt 明確限制「只能根據 evidence 作答」

這份作業除了主 pipeline，也補了完整的工程化能力：

- 實作本地評估腳本 [HW2/score_public.py](./HW2/score_public.py)，以 ROUGE-L 重現助教 evidence score 的計算方式
- 撰寫 [HW2/ablation_study.py](./HW2/ablation_study.py) 做 chunk size、retrieval、reranker 與 evidence selection 的系統化調參
- 保存逐階段 log，方便回看 chunking、retrieval、reranking、LLM generation 的行為與瓶頸
- 在輸出前做 submission format validation，避免格式錯誤影響評分

這份作業展現的能力：

- RAG pipeline design
- Retrieval / reranking / prompt orchestration
- Evidence-grounded generation
- 實驗設計、ablation study 與評估工具實作

目前 repo 中可直接看到的成果：

- `HW2/outputs/ablation_results_v1.csv` 顯示在 20 篇 public sample 上，最佳 chunking 組合的平均 evidence score 為 **0.2554**
- `HW2/submitted_file/cor_50_evi_0.2798.json` 保留了一個公開測試版本快照；依檔名可讀出該版本 evidence score 為 **0.2798**

## Tech Stack

`Python`, `PyTorch`, `Transformers`, `Unsloth`, `LoRA`, `Sentence-Transformers`, `FAISS`, `BM25`, `OpenRouter API`, `ROUGE-L`, `pandas`, `scikit-learn`

## Repository Guide

- [HW1](./HW1): 病理學選擇題 QA，重點在小模型微調與 prompt 設計
- [HW2](./HW2): 論文文件問答系統，重點在 RAG pipeline 與 retrieval experimentation

註：
- HW1 的 71.11% 來自 notebook 中保存的 validation output。
- HW2 的 0.2554 來自 `ablation_results_v1.csv`。
- HW2 的 0.2798 來自已保存提交檔名 `cor_50_evi_0.2798.json`，此處依檔名做結果標示。
