# HW1: Pathology QA with Llama-3.2-1B + LoRA

這份作業的目標是解決**病理學單選題問答**：輸入題目與四個選項，模型只輸出正確答案 `A / B / C / D`。  
目前主要實作已整理成 [main.py](./main.py)，使用 `unsloth/Llama-3.2-1B-Instruct` 搭配 **4-bit quantization、LoRA fine-tuning、few-shot prompt、資料增強、hard example mining 與第二階段再訓練**，完成一個可訓練、可驗證、也可輸出 submission 的 medical MCQ pipeline。

## Project Summary

| Item | Content |
| --- | --- |
| Task | Medical pathology multiple-choice question answering |
| Base model | `unsloth/Llama-3.2-1B-Instruct` |
| Fine-tuning | 4-bit + LoRA + 2-stage training |
| Prompt strategy | Few-shot + strict output format + reasoning-oriented instruction |
| Training style | Supervised fine-tuning + data augmentation + hard example retraining |
| Latest result | Hold-out split accuracy **74.86%** |

## Dataset

專案使用兩份 CSV：

| File | Description | Size |
| --- | --- | --- |
| [dataset/dataset.csv](./dataset/dataset.csv) | 有標註答案的訓練資料 | 9000 題 |
| [dataset/benchmark.csv](./dataset/benchmark.csv) | 無標註答案的測試資料，用於產生提交檔 | 900 題 |

資料欄位如下：

- `question_id`: 題目編號
- `question`: 問題文字
- `opa`, `opb`, `opc`, `opd`: 四個選項
- `ans`: 正確答案索引，`0/1/2/3` 對應 `A/B/C/D`

在目前 `main.py` 版本中，`dataset.csv` 會先被切成：

- Train: 7200
- Validation: 900
- Test: 900

註：程式輸出顯示為 `Validation Accuracy`，但實際推論迴圈跑的是 `test_df`，也就是內部分出的 hold-out 測試切分。

## Method

### 1. Base Model and PEFT

模型使用 `unsloth/Llama-3.2-1B-Instruct`，並透過 Unsloth 載入 4-bit 量化版本以降低 VRAM 需求。  
微調時使用 LoRA，主要設定如下：

- `r = 256`
- `lora_alpha = 512`
- `lora_dropout = 0.03`
- target modules:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- `modules_to_save = ["embed_tokens", "lm_head"]`

這樣的設計可以在較小資源下完成指令微調，同時保留不錯的任務適應能力。

### 2. Prompt Design

訓練與推論都使用固定模板：

- system prompt 將模型設定為 medical pathology expert
- 明確要求模型**只能輸出單一字母**
- 放入數個 few-shot 範例題
- 在指令中加入 step-by-step reasoning 導向，但最後輸出仍限制為 `A/B/C/D`

這個 prompt 設計的重點不是讓模型輸出解釋，而是引導模型先進行病理判斷，再把最終輸出壓縮成競賽需要的單字母格式。

### 3. Data Preparation

資料處理流程如下：

- 將答案從 `0/1/2/3` 轉成 `A/B/C/D`
- 建立全域選項池，把所有醫學選項文字收集成 `global_option_pool`
- 對訓練集做資料增強：保留正確答案，另外從全域選項池隨機抽 3 個 distractors 重組選項
- 將原始訓練集與增強後訓練集合併，形成更大的訓練資料
- 轉成 Hugging Face `Dataset` 並格式化成完整 prompt

在目前版本中，train split 會先從 7200 題擴增成 **14400** 筆訓練樣本。

### 4. Training Objective

第一階段訓練採用 `SFTTrainer`，並使用 `DataCollatorForCompletionOnlyLM`，只對 `[Answer]:` 之後的輸出計算 loss。  
這讓模型主要學習「在看到完整題目與選項後，輸出正確答案字母」。

第一階段主要訓練設定：

- `max_seq_length = 1024`
- `per_device_train_batch_size = 8`
- `gradient_accumulation_steps = 16`
- `num_train_epochs = 2`
- `learning_rate = 3e-5`
- `optim = "adamw_8bit"`
- `lr_scheduler_type = "cosine"`
- `neftune_noise_alpha = 5`
- `eval_steps = 20`

### 5. Hard Example Mining and Stage 2 Retraining

第一階段完成後，程式會在原始 `train_df` 上做一輪推論，把模型答錯的題目視為 hard examples。

第二階段資料組成方式：

- hard examples 重複兩次，提高權重
- 從答對的 easy examples 中抽取最多 `3x hard examples` 作為 anti-forgetting buffer
- 合併後重新 shuffle，形成 stage 2 訓練集

第二階段再訓練設定較保守：

- `per_device_train_batch_size = 2`
- `gradient_accumulation_steps = 4`
- `num_train_epochs = 2`
- `learning_rate = 5e-7`

這個設計的目的是讓模型集中修正錯題，同時降低 catastrophic forgetting 的風險。

### 6. Validation and Error Analysis

驗證時做了幾個實用的工程處理：

- 開啟 `FastLanguageModel.for_inference(model)` 提升推論效率
- `max_new_tokens = 5`，因為目標只是一個答案字母
- 只解碼**新生成的 tokens**，避免 few-shot 範例中的答案被誤抓
- 用 regex 抓第一個合法的 `A/B/C/D`
- 若模型沒有輸出合法字母，fallback 為 `A`
- 將答錯題目輸出成 `val_error_report.csv`，方便後續分析

另外，程式也會輸出：

- answer distribution plot
- training / validation loss 與 learning rate 圖
- confusion matrix 與 validation metrics 圖

### 7. Benchmark Submission Inference

對 `benchmark.csv` 產生最終答案時，程式使用**高溫多數決推論**：

- 每題做多次 sampling
- 從多個輸出中解析 `A/B/C/D`
- 以票數最高的答案作為最終提交結果

這個設計的目的是降低單次生成的不穩定性，讓最終 submission 更穩定。

## Result

目前 `main.py` 版本的 hold-out split 結果如下：

| Metric | Score |
| --- | --- |
| Accuracy | **74.86%** |

相較於前一版 notebook 保存的 `71.11%`，新版腳本透過資料增強與 hard example retraining，進一步把 hold-out accuracy 提升到 **74.86%**。

## How To Run

這份作業目前已整理成 **script-first workflow**，最適合在有 GPU 的環境執行，例如 Google Colab、Linux + CUDA，或本機 GPU 環境。

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

如果你直接使用 notebook 內的安裝 cell，也可以依照其中的版本安裝 `unsloth`、`trl`、`transformers` 等套件。

### 2. Run the script

在 `HW1` 目錄下執行：

```bash
python main.py
```

腳本會依序執行：

1. 載入模型與 tokenizer
2. 切分資料集
3. 建立 augmentation dataset
4. 第一階段 fine-tuning
5. hold-out 驗證與錯誤報告輸出
6. hard example mining
7. 第二階段再訓練
8. benchmark submission 產生

### 3. Update paths if needed

在 [main.py](./main.py) 開頭可調整：

- `dataset_path = "./dataset/dataset.csv"`
- `benchmark_path = "./dataset/benchmark.csv"`
- `dir_path = "./"`
- `try_name = "fine_tune_4"`

目前程式會把輸出集中存到 `./fine_tune_4/` 底下。

### 4. Generate submission

執行完成後，會輸出：

- `fine_tune_4/submission.csv`
- `fine_tune_4/val_error_report.csv`
- `fine_tune_4/plots/answer_distribution.png`
- `fine_tune_4/plots/training_metrics.png`
- `fine_tune_4/plots/val_metrics.png`

## Repository Structure

```text
HW1/
├── dataset/
│   ├── dataset.csv
│   └── benchmark.csv
├── main.py
├── main.ipynb
├── README.md
└── requirements.txt
```

## What This Project Demonstrates

- 使用小型 LLM 做 domain-specific instruction tuning
- 以 LoRA 完成 parameter-efficient fine-tuning
- 透過 prompt engineering 控制輸出格式與任務行為
- 透過資料增強改善選項偏差與泛化能力
- 以 hard example mining + stage 2 retraining 修正模型弱點
- 針對多選題任務設計實用的 inference / post-processing 流程
- 使用 accuracy、錯誤報告、混淆矩陣與訓練曲線做模型分析

## Possible Next Improvements

- 修正 benchmark majority voting 區塊，將推論參數與函式介面整理成一致版本
- 把 stage 1 / stage 2 checkpoint 與最佳模型選擇流程獨立整理
- 針對錯誤較多的類別做 targeted prompt / curriculum 調整
- 比較不同 few-shot 範例數量與 LoRA 設定對結果的影響
