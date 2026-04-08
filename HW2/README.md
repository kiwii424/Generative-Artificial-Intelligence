執行指令：

```bash
# 切到 HW2 目錄
cd /Users/meredithhuang/Desktop/code/Python/GAI/HW2

# ① 先在 public dataset 測試（會顯示 ROUGE-L 分數）
uv run python 111511157.py --eval --output public.json

# ② 跑完後用官方腳本驗證（需要先啟動 vLLM server）
uv run python score_public.py public.json --port 8091 --host 192.168.0.7

# ③ 最終提交：跑 private dataset
uv run python 111511157.py
```

`--eval` 模式會印出每個階段的詳細資訊：

* **Chunking**：切了幾個 child chunk、平均幾 chars
* **Indexing**：FAISS vectors 數量
* **Query**：三個查詢變體是什麼
* **Retrieval**：dense/BM25 候選數、RRF 候選、rerank 分數
* **LLM**：傳了幾個 parent chunk、估計 token 數、實際用了幾 token
* **Result**：答案內容、evidence 數量



A. 產出正式作業檔案 (針對 Private Dataset)

當你要生成最終交給助教的** **`111511157.json` 時，直接執行：

**Bash**

```
uv run python 111511157.py
```

* **用途**：讀取** **`private_dataset.json`，跑完 100 題，並存檔。
* **預設路徑**：它會去找同資料夾下的** **`private_dataset.json`。

#### B. 測試效能與跑分 (針對 Public Dataset)

如果你想知道目前這套演算法的分數，是否超過助教的 Strong Baseline (0.26185)：

**Bash**

```
uv run python 111511157.py --eval
```

* **用途**：讀取** **`public_dataset.json`。
* **特別之處**：跑完後會自動計算** ****ROUGE-L 分數**，並告訴你目前的成績落點。這對調教參數（例如改 Chunk 大小）非常有幫助。

#### C. 指定特定的資料集檔案

如果你把檔案放在不同資料夾，或是想跑自訂的測試檔：

**Bash**

```
uv run python 111511157.py --dataset my_test_data.json
```

---

### 2. 不同參數 (Arguments) 的詳細差別


| **參數**    | **全名**  | **功能說明**                  | **影響**                                                             |
| ----------- | --------- | ----------------------------- | -------------------------------------------------------------------- |
| **無參數**  | (Default) | 跑**Private** 模式。          | 生成正式上傳用的`.json` 檔。                                         |
| `--eval`    | Evaluate  | 跑**Public** 模式並計算分數。 | **不會**讀取 private 檔，而是讀取有標準答案的 public 檔來評分。      |
| `--dataset` | Dataset   | 指定輸入檔案的路徑。          | 強制程式讀取你指定的`.json`，無視預設路徑。                          |
| `--output`  | Output    | 指定輸出的檔名。              | 預設是`111511157.json`，你可以改成 `test_v2.json` 以免覆蓋掉舊結果。 |

---

### 3. 這套 Pipeline 的運作流程圖

了解執行指令後，這份程式碼在跑的時候其實做了以下幾件事：

1. **Chunking (分塊)**：將長論文切成 200 字的小塊（Child，為了高 ROUGE-L 分數）以及 800 字的大塊（Parent，為了給 LLM 完整語境）。
2. **Hybrid Search (混合檢索)**：同時用 FAISS (語意) 和 BM25 (關鍵字) 去找資料。
3. **RRF Fusion**：把兩種搜尋結果合併，取長補短。
4. **Reranker (重排序)**：用更強的模型重新檢查這 20 個結果中，誰跟問題最相關。
5. **LLM Generation**：把選出的 Parent Chunks 丟給 Llama-3.2-3B 生成最終答案。

---

### 4. 執行前的最後檢查

* **確認 API Key**：確保** **`.env` 檔案裡有** **`OPENROUTER_API_KEY=sk-or-v1-xxxx...`。
* **確認檔案位置**：確定** **`private_dataset.json` 或** **`public_dataset.json` 放在跟** **`.py` 同一個資料夾。
* **第一次執行的等待**：第一次執行時，程式會下載大約 2GB 的模型權重（BGE Embedding & Reranker），這取決於你的網速。下載完後，之後執行就會直接載入。

**建議先跑跑看** **`uv run python 111511157.py --eval`，看看你的電腦跑一題平均要幾秒！** 如果覺得跑太慢或噴錯，隨時把 Log 貼給我。
