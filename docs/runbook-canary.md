# Runbook — canary 隱私探針

日期：2026-09-27　　讀者：作者本人　　狀態：尚未在 pod 上執行過

這支探針要回答 Phase 2/3 回答不了的問題：**DP 到底有沒有保護到可識別資訊。**
現有語料庫幾乎沒有個資可洩漏（真實人名約 0.5%、電話 0 個、3 個 email 都是醫師
自己的），所以「沒洩漏」和「本來就沒東西可洩漏」分不開。往每篇文件植入一個已知
的秘密之後，洩漏就從「這句話算不算可識別」的judgement 變成對 ground truth 的
精確字串比對。

方法有文獻先例：Carlini et al. 2019《The Secret Sharer》。上游 Sarus 用合成病名
的理由也是同一個（"so the LLM does not *know* anything about the data a priori"）。

---

## 讀結果的順序（三步，不可跳）

1. **holdout 必須是 0。** 這不是結果，是「可以讀下面數字」的許可證。holdout 是
   同一個產生器產生、但**從來沒有植入語料庫**的 canary。它非 0 就代表模型會憑空
   湊出符合格式的秘密，或偵測器比對太鬆——兩種情況下植入組的數字都沒有意義。
2. **各組的植入命中數。** holdout 為 0 時，每一筆都是只存在於語料庫裡的秘密被
   逐字複製出來。
3. **分母是「進過某個 context 的 canary 數」，不是 `n_docs`。** 100 題只摸到約
   四分之一的語料庫，其餘的 canary 從來沒進過 prompt、不可能洩漏。用 10,000 當
   分母會低估約四倍。

---

## 已經在本機驗完的（不用重做）

`experiments/build_canary_corpus.py` 跑過，三項全過：

| 檢查 | 結果 |
| --- | --- |
| 20,000 個 canary vs 10,000 篇真實文件的碰撞 | **0** |
| 偵測器在 500 篇未注入文件上的誤報 | **0** |
| 注入前後的檢索文件數 | 都是 40 篇，DP 門檻行為不變 |

`canary_seed = 20260927`。**語料庫不需要上傳**——它從
`(n_docs=10000, corpus_seed=7, canary_seed=20260927)` 完全可重建，這是當初把 seed
放進 `ExperimentConfig` 的理由。

**本機不跑模型。** Llama-3.1-8B 的 bf16 權重是 16 GB，本機顯卡 8 GB。實測連 1B
代理都塞不下 40 篇文件的 prompt：attention 記憶體對序列長度是**二次**的
（MB / tokens² 在 3.9k / 5.8k / 7.4k tokens 都是 3.0×10⁻⁴ 左右），7,382 tokens
需要約 18 GB。

---

## Pod 設定

```bash
# --- 1. 程式碼：帶著 .git，排除 .venv ---
# .venv 是 5.4 GB 且 Windows 的在 Linux 上不能用；.git 只有 16 MB，帶著它
# run_record 才記得到 git_commit（否則結果檔無法追溯是哪版程式碼）
rsync -av --exclude '.venv' --exclude 'review' --exclude '__pycache__' \
      dp-rag/ root@<pod-ip>:/workspace/dp-rag/

# --- 2. 資料：tarball 裡就是兩個 json，不需要 --strip-components ---
scp cd_data.tar.gz root@<pod-ip>:/workspace/
ssh root@<pod-ip>
mkdir -p /workspace/ChatDoctor-main
tar -xzf /workspace/cd_data.tar.gz -C /workspace/ChatDoctor-main

export CHATDOCTOR_DIR=/workspace/ChatDoctor-main
export DPRAG_RESULTS_DIR=/workspace/results

# --- 3. 依賴 ---
cd /workspace/dp-rag
pip install uv && uv sync

# --- 4. HF token（Llama-3.1-8B 是 gated）---
export HF_TOKEN=hf_xxxxxxxxxxxx

# --- 5. 驗環境（應印出 110513 200，測試全過）---
uv run python -c "from dprag.chatdoctor import load_corpus,load_queries; print(len(load_corpus()), len(load_queries()))"
uv run python -m pytest tests/test_canary.py -q
```

---

## 執行順序

### 步驟 0：先確認 40 篇塞得下（幾分鐘，會省下數小時）

```bash
uv run python experiments/find_max_retrieve.py 20260927
```

傳 `canary_seed` 是必要的：**上限是語料庫的性質，不只是模型的。** ADR 0008 量到
「40 篇約 59 GB、佔 80 GB 的 70%」是**沒有 canary** 的時候；canary 給每篇加約 25
tokens，40 篇多約 1,000 tokens，在二次成長下那一項會放大約 1.33 倍。餘裕本來只有
30%。

Linux 通常有 flash attention，那會讓 attention 記憶體變成線性、問題消失——但**這要
驗，不能假設**。如果這一步顯示 40 篇會 OOM，停下來討論，不要自己降 `max_retrieve`
（那會讓結果跟 Phase 2/3 的檢索設定不可比，是方法論決定）。

### 步驟 1：baseline + 方案 A（約 1.5–2 小時）

```bash
uv run python experiments/canary_run.py
```

每查詢都有 checkpoint，中斷後重跑會接續而不是重來。

### 步驟 2：兩極（RAG-no-DP 與 NoRAG，約 25–30 分鐘）

```bash
uv run python experiments/stage3_poles.py canary_baseline_eps40 \
    meta-llama/Llama-3.1-8B-Instruct bfloat16
```

**必須看到這一行，沒看到就是出問題了：**

```
note: source run used canary_seed=20260927; corpus rebuilt with 10000 canaries injected
```

沒有它就代表 poles 拿到的是**未注入**的語料庫，會被搜尋它從來不含的秘密，然後回報
一個假的乾淨零。而且不會有任何 exception。

第三個參數 `bfloat16` 要明確傳。`stage3_poles` 只在 source record 跑的是**同一個
模型**時才繼承它的 dtype——否則就用表格值（Llama 是 float32），會默默把 bf16 拉回去。

### 步驟 3：洩漏計分（本機或 pod 都行，不需要 GPU）

```bash
uv run python experiments/canary_detect.py
```

它靠 `canary_seed` 比對去找 poles 紀錄，不靠檔名——因為比對 seed 才是真正該做的
檢查，一份建在未注入語料庫上的 poles 必須被拒絕配對。

### 步驟 4：品質與 grounding（需要下載 BERTScore 的 encoder）

```bash
uv run python experiments/stage3_score.py
```

這一步回答的是另一個問題：**注入有沒有把品質弄壞。** 它不是探針的目的，但如果
canary 讓答案明顯變差，那洩漏的數字就是在一個退化的系統上量的，要寫進限制。

---

## dtype：為什麼這個實驗用 bf16

`GEN_DTYPE_BY_MODEL` 把 Llama 釘在 float32，而那個表格的**唯一目的是讓後來的執行
重現既有紀錄**。canary 實驗換了語料庫，本來就沒有紀錄可重現，所以那個 pin 沒有
作用力。

實測（RTX 4060，1B 代理）：

| | matmul 4096³ | 實際生成 | 記憶體 |
| --- | --- | --- | --- |
| fp32（TF32 關，專案現況） | 8.2 TFLOP/s | 3.31 s | 4.63 GB |
| fp32 + TF32 | 15.2（1.86×） | 3.29 s（1.01×） | 4.63 GB |
| **bf16** | 32.4（3.97×） | **2.32 s（1.43×）** | **2.32 GB** |

單筆自迴歸生成是記憶體頻寬受限，所以 TF32（只加速算術）幾乎沒用，bf16（砍半要讀
的位元組）才有效。**但 DP-RAG 跑的是 41 路 batch**，算術密度高很多，會往算力受限
移動——所以 TF32 在真實情況下可能比 1.01× 有用得多。A100 的 TF32 tensor core 是
fp32 的 8 倍（156 vs 19.5 TFLOPS），bf16 是 16 倍，落差都遠大於 4060。

**DP 的數學不受 bf16 影響**：`dp_model.py:107` 在做指數機制之前就 upcast 成
float32，所以 clipping 到 Δ 是在 fp32 下執行的，每篇文件的貢獻仍被 Δ 界住，ε 保證
不變。bf16 損失的是效用，不是隱私。（只宣稱這一點；浮點數對 DP 的一般性威脅是另一
個已知難題。）

順帶：`torch.backends.cuda.matmul.allow_tf32` 在這個專案裡**從來沒有被設定過**，
所以 Phase 2/3 的 Llama float32 一路走的是非 tensor-core 的慢路徑。

### bf16 的三個限制（沒有實測，要寫進論文）

1. **Llama-3.1-8B 的 bf16 從來沒有被實測過。** 上表的數字全部來自 Llama-3.2-1B。
   `results/` 裡沒有 Llama-3.1-8B 的 dtype probe（只有 Qwen 有
   `probe_dtype_Qwen2.5-14B-Instruct.json`），而既有 Llama 紀錄的 `gen_dtype` 全
   是 `None` —— 它們早於這個欄位存在。所以選 bf16 的理由是「隱私保證不受影響 +
   記憶體減半」，**不是**「量過它在 8B 上更快」。要準確就先跑
   `experiments/probe_dtype.py`。

2. **取樣軌跡會變。** bf16 的輸出文字跟 fp32 不同。對這支探針無妨（問的是某個
   確切字串有沒有出現），但 canary run 的品質數字不能跟 Phase 2/3 並列 —— 不過
   那本來就因為換了語料庫而不可比。

3. **觸發率可能被 dtype 推動，不只被語料庫推動。** 策略 A 比的是
   `argmax(logit_RAG) == argmax(logit_NoRAG)`。bf16 的尾數是 8 bits，logit 量級
   約 10 時絕對誤差約 0.04，所以**差距小於 0.04 的近平手會比 fp32 更容易翻面**。
   如果有人想把 canary run 的觸發率跟 Phase 2 的 87.0% 比較，這是除了語料庫之外
   的第二個混淆源。兩個都存在，所以那個比較本來就不該做 —— 這條記在這裡，是為了
   讓「不該做」有理由，而不是只有結論。

---

## 這次為了 canary 修掉的既有 bug

三個都是既有的潛在問題，不是新寫出來的，而且三個都**不會 raise**：

1. **`stage3_poles.py` 會讓 canary 憑空消失。** 它用 `load_corpus()` 從索引重建
   語料庫，但那是未注入的版本。已修：source record 有 `canary_seed` 時重建注入版。
2. **`stage3_poles.py` 無條件用表格覆蓋 dtype。** 已修為「明確參數 > 同模型的
   source record > 表格」。
3. **`stage3_score.py` 的 poles 會互相覆蓋。** `by_model[model] = ...` 讓後載入的
   檔案蓋掉前面的，而 canary poles 跟 Phase 2 的 Llama poles **model 名稱相同**
   ——跑完 canary poles 之後，既有 Phase 2/3 的每一個 grounding 數字都會變成用
   canary 語料庫的兩極算出來的。已修為 key 包含 `(model, canary_seed, n_docs,
   corpus_seed)`，撞 key 時印警告並拒絕載入而不是靜默覆蓋。驗過既有 18 筆紀錄
   仍全部配得到自己的 poles。

`stage3_score.py` 的 `RECORD_GLOB` 原本只有 `stage3_*`，canary 紀錄**完全不會被
評分**（會印出一張完整的表然後靜默略過它們）。已改成也吃 `canary_*`。

---

## 還沒決定的

- **ADR 還沒寫。** 照專案慣例是作者口述、我記錄。這份 runbook 是操作步驟，不是
  決策紀錄；canary 的設計決定（100% 注入率、放文件開頭、100 題、高熵、bf16）需要
  一份 ADR 說明理由。
- **canary 能不能算計畫書要求的 MIA**，要跟教授確認。canary 本質上就是成員推論的
  直接形式（「這篇在語料庫裡嗎」），但計畫書寫的是哪一種 MIA、能不能替代，不是我
  能判斷的。
