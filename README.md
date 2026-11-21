# NCM-GPT1

以 GPT-1 體量實作的 **Next-Concept Modeling（NCM）**：先把文字壓成多層離散概念碼，再由粗到細、同一層一次並行預測。這**不是** 2018 年 GPT-1 權重的複製品，也不是下一個 token 的語言模型；transformer 軀幹沿用 GPT-1 的尺寸（12 層、寬度 768、12 個注意力頭、FFN 3072、上下文 512），訓練目標改為論文的兩階段 NCM。

模型本體（`models/`、`tokenization/`、`utils/`）直接沿用 [HazyResearch/ncm](https://github.com/HazyResearch/ncm)（MIT），本專案補上官方倉庫沒有的訓練流程與 CPU 煙霧測試。論文：[arXiv:2610.14162](https://arxiv.org/abs/2610.14162)。

## 跟 GPT-1 差在哪裡

| | 2018 GPT-1 | 這個 NCM（`config/gpt1.yaml`） |
|---|---|---|
| 預測單位 | 下一個 token，由左到右 512 步 | 概念碼階層，由粗到細 **23 層** |
| 同層內 | 序列相依 | **同一層的碼全部並行** |
| 第 0 層 | — | 不由 transformer 預測，推論時從訓練 unigram 取樣 |
| Tokenizer | BPE（約 4 萬詞） | 多尺度殘差 VQ-VAE（先訓練、再凍結） |
| 解碼 | 直接是 token | VQ decoder 對最細層做 argmax（與官方評測相同） |

無條件生成：先抽 level-0 碼 → transformer 依序生成其餘 22 層（層內並行）→ decoder 還原成 512 個 token。生成步數是 23，不是 512。

## 架構與參數數量

數字是在 CPU 上實例化官方模組後量到的，沒有做完整訓練。

| | 煙霧測試 `smoke.yaml` | 完整 `gpt1.yaml` |
|---|---:|---:|
| 上下文 L | 16 | 512 |
| 層數 K、層寬 | 4 層：`[1, 4, 8, 16]` | 23 層：完美平方到 512，`[1, 4, 9, …, 484, 512]` |
| 碼本大小 | 每層 64 | 每層 4096 |
| Tokenizer | `embed_dim=32`，4 頭，`enc_layers=0`，`dec_layers=1` | `embed_dim=384`，6 頭，同上深度 |
| 表層詞表 | 128（內建字元，不必上網） | 50304（gpt2 + `<|pad|>`，與官方 OWT 對齊） |
| Generator | 2 層 × 32 維 × 4 頭 | 12 層 × 768 維 × 12 頭 |
| 前綴 | 長度 1（EOS） | 長度 1（EOS），`raw_token_prefix=true` |
| **Tokenizer 參數** | **58,576（0.06M）** | **92,276,480（92.28M）**；非嵌入 72.76M |
| **Generator 參數** | **32,992（0.03M）** | **101,688,854（101.69M）**；非嵌入 101.38M |

Generator 少於原始 GPT-1 的約 1.17 億，是因為沒有 4 萬詞的 token 嵌入表（改為把 VQ 維度 384 投到 768，輸出頭與碼本綁定）。開啟與官方 OWT 相同的 `post_upsample_conv` 後，生成器約為 101.69M。Tokenizer 是額外的模組，裡面含 23 本 4096×384 的碼本與兩張 50304×384 的表層嵌入。

## 安裝（CUDA GPU）

需要 Python 3.10+ 與 NVIDIA 驅動。完整 12×768 模型請在 GPU 上訓練。偵測到 CUDA 時，訓練會自動使用混合精度（支援 bf16 的卡用 bf16，否則用 fp16 + GradScaler）並嘗試 `torch.compile`。可用 `--no-amp` 或 `--no-compile` 關閉。

Windows：

```bat
install_cuda.bat
python scripts\check_cuda.py
```

Linux：

```bash
python3 -m venv .venv && source .venv/bin/activate
bash install_cuda.sh
```

腳本會先裝 CUDA 版 PyTorch（優先 cu128，失敗再試 cu126），再裝其餘套件。只跑 CPU 煙霧測試時，改裝 CPU wheel：`pip install torch --index-url https://download.pytorch.org/whl/cpu` 然後 `pip install -r requirements.txt`。

在專案根目錄執行腳本，或設定 `PYTHONPATH` 為本目錄（套件以 `from models...` / `from tokenization...` / `from utils...` 匯入）。

## 兩階段訓練

**第一階段**訓練多尺度殘差 VQ-VAE。損失是 token 交叉熵（忽略 PAD）加上量化損失；解碼用 argmax 對 token 算重建正確率。

**第二階段**凍結 tokenizer，訓練 12×768 的 `NextConceptTransformer`：先 `encode` 出乾淨碼當標籤，再按層對**輸入**碼做均勻替換（腐蝕率由粗到細線性遞減，平均 0.50），以 `vq.idx_to_ncm_input` 組輸入。損失權重

\[
w_k = \ell_k^{-\alpha} \big/ \sum_j \ell_j^{1-\alpha}
\]

預設 `alpha=0`，每個碼位置權重相同，對應論文的 ELBO。checkpoint 內的 `inference.level0_unigram` 是訓練集 level-0 碼的正規化頻率，給 `generate.py` 與官方 `eval_shard.py` 使用。

### 煙霧測試（CPU，數秒，不需網路）

```bash
./tests/run_smoke.sh
# 或分開跑：
python tests/test_ncm_gpt1.py
python train_tokenizer.py --config config/smoke.yaml --data data/smoke_corpus.txt --max_steps 8
python train_generator.py --config config/smoke.yaml --data data/smoke_corpus.txt \
    --tokenizer_ckpt artifacts/tokenizer/tokenizer.pt --max_steps 8
python generate.py --config config/smoke.yaml --n 2 --seed 0 --show_levels
```

品質會很差，這是預期的；只要行程結束碼為 0 即可。

### 完整 GPT-1 設定（需要 GPU，不要在這台機器上開跑）

```bash
# 建議資料：TinyStories，或本機 utf-8 純文字（空行分段）/ jsonl（text 欄）
python train_tokenizer.py --config config/gpt1.yaml \
    --hf_dataset roneneldan/TinyStories
python train_generator.py --config config/gpt1.yaml \
    --tokenizer_ckpt artifacts/tokenizer/tokenizer.pt \
    --hf_dataset roneneldan/TinyStories
python generate.py --config config/gpt1.yaml --n 4 --temperature 1.0 --show_levels
```

- `--data path/to/corpus.txt` 與 `--hf_dataset` 可並用。`--hf_dataset` 需要可選套件：`pip install datasets`。
- 預設 batch size 為 16（約 16 GiB 顯存、bf16）。若發生 OOM，改傳 `--batch_size 8`；顯存還有餘裕可試 32。
- 原始 GPT-1 用的是 BooksCorpus。開源接近項是 `lucadiliello/bookcorpusopen`（Kaggle 預設）；也可用 TinyStories 或自備文本。HuggingFace 資料集改為串流讀取，避免一次載入整份語料。
- 可 `--resume` 或 `--resume auto` 續訓（會找 `latest.pt`）。檢查點是先寫暫存再替換，並存下 GradScaler；`torch.compile` / DDP 的權重在存檔時會先解開。中間 step 檔只保留最新兩個，以免把磁碟塞滿。
- 多卡：`torchrun --nproc_per_node=2 train_tokenizer.py ...`（generator 同理）。batch size 是**每張 GPU** 的大小。VQ 碼本 EMA 已跨卡同步，死碼重啟只在 rank 0 執行再廣播。
- 日誌每 `--log_every` 步（預設 50）印損失、速度（steps/s、tokens/s）、顯存，並寫入 `*_log.jsonl` 與 `status.json`。可用 `--max_hours` / `--reserve_minutes` 在時限前存檔退出，`--max_windows` / `--max_docs` 限制打包量。
- 有 CUDA 時預設開混合精度，不必再加 `--amp`。`--no-amp` 可改回 fp32；`--device cpu` 可強制用 CPU。
- 有 CUDA 時預設嘗試 `torch.compile`（失敗會自動退回一般模式）。`--no-compile` 可關閉。Kaggle 上預設關閉 compile。
- 本機 `gpt1.yaml` 開啟 Flex Attention。若啟動失敗，把該項改回 `false`，會改走等價的分層 SDPA。
- 生成前會先用「句 → 片語 → 詞」切開（`splitter`，預設規則切分）。訓練時視窗會盡量在句界開始；生成展開時，粗層對應句子、中層對應片語、細層對應詞，只細化仍不確定的單位。可用 `python scripts/diagnose_split.py --file data/smoke_corpus.txt` 查看切分樹。若要改用小型神經切分器，跑 `python train_splitter.py --config config/gpt1.yaml`，再把 `splitter.backend` 設成 `model`。
- 預設「只在需要時才展開」：生成時若目前解碼已經夠確定且穩定，就提前停掉後面的層；同一層裡高把握的片段（類似圖像的黑背景）不再細化，改加接近 0 的殘差。可用 `--no-adaptive` 關閉。tokenizer 訓練也會依殘差能量跳過過弱的 cell。可用 `python scripts/diagnose_expand.py --tokenizer_ckpt ...` 查看每一層的殘差與把握度。

無條件前綴是一顆 GPT-2 EOS（shape `(B, 1)`）。`generate.py --show_levels` 會把每一層的中間重建印出來，方便看由粗到細的過程。


## 在 Kaggle 用約 30 GPU 小時訓練（T4 × 2）

Kaggle 每週約 30 個 GPU 小時，單場最長 12 小時，T4 × 2 共 32 GiB（兩張 16 GiB，不是一張 32 GiB）。T4 沒有硬體 bf16，訓練走 fp16 + GradScaler。請開 Internet，並用 **Save & Run All**。

1. 上傳 [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb)，Accelerator 選 **GPU T4 x2**。
2. notebook 會 clone 本倉庫、安裝 `omegaconf hydra-core datasets regex`（**不要重裝 torch**），然後執行：

```bash
python scripts/train_kaggle.py --hours 11.5 --reserve_minutes 15 \
    --output_dir /kaggle/working/artifacts --config config/kaggle.yaml \
    --batch_size 2 --gradient_checkpointing
```

3. 編排腳本先看 tokenizer 的 `status.json`。`finished: true` 就跳過，再用剩餘時間訓練 generator。時間到會存 `latest.pt` 並退出。v8 正在跑，不要中斷。v8 結束後再掛上 **COMPLETE 的 kernel 第 8 版 Output**（tokenizer 已完成；generator 從 v8 的 `latest.pt` 續，v8 是從 step 40457 接著跑的）。
4. 預設語料是 BookCorpus 開源重建版；若下載失敗，會改用 MiniPile，再不行才用 TinyStories。`config/kaggle.yaml`：Flex 關閉、chunked attention、梯度檢查點、T4 用 fp16、generator 每卡 **batch 2**、`max_windows: 524288`（約 2.68 億 token，並快取 `artifacts/packed_windows.pt`；其他 rank 輪詢這個檔，不在 NCCL barrier 上等打包）、`alpha: 0.5`、`mid_aux: 0.5`。v7 把 batch 從 2 加到 4 之後，tokens/s 仍約 1400，steps/s 從約 0.69 降到約 0.34，所以步數預算用 batch 2 更快。`alpha` 與 `mid_aux` **都不是**論文 ELBO（本機 `gpt1.yaml` 仍是 `alpha: 0`、沒有 `mid_aux`）。v7 在 `alpha: 0.5` 下，最短預測層 CE 約 4.5、最細 512 層約 1.5，中間長度 81–225 仍約 5.2–6.1；`mid_aux` 把該段的平均 CE 再加進總損失。總損失不要和 v5（alpha 0）或 v7/v8（alpha 0.5、mid_aux 0）比較；各層 CE 仍可比較。每 1000 步在 rank 0 對 256 個保留視窗算 val loss 與各層 CE。
5. 日誌在 `artifacts/tokenizer/tokenizer_log.jsonl` 與 `artifacts/generator/generator_log.jsonl`。舊欄位（`loss`、`ce_first` / `ce_mid` / `ce_last`、`ce`、`acc`）都還在，另外多了 `ce_levels`、`grad_norm`、`data_epoch`、`corrupt_p`、`mid_aux`，tokenizer 另有 `fine_drop_keep`。`status.json` 的 `finished: true` 代表該階段已跑完設定的 `max_steps`。規則切分器沒有權重，會寫 `artifacts/splitter/status.json`，並照樣按句打包。

v8 的權重與 optimizer 會從該檢查點載入。下一場的總損失多了 `mid_aux`，不要和 v8 的 loss 曲線直接比較。

梯度檢查點維持開啟。v7 在 batch 4 時峰值約 4.1 GiB，沒有 OOM；改回 batch 2 是為了每小時更多步，不是因為顯存不夠。

多卡本機等價指令：

```bash
torchrun --nproc_per_node=2 train_tokenizer.py --config config/kaggle.yaml \
    --hf_dataset lucadiliello/bookcorpusopen --resume auto --max_hours 11.5 --no-compile
```

## 官方程式碼與已知細節

- 核心模型**沒有重寫**，只多註冊了一個煙霧測試用的字元 tokenizer（`tokenization/char_tokenizer.py`）。
- 官方 `VQVAE.reconstruct` 曾把 `vq()` 解包成 3 個值（`forward` 實際回傳 5 個）；本專案已改成安全解包。
- 與官方 OWT 文字設定對齊：`post_upsample_conv` 開啟（零初始化、可學習）、綁定碼本輸出頭時也開 `tie_heads_bias`。
- 生成器序列長度是前綴 + 各層碼（約 4307），不是 512。訓練時預設用**分層 SDPA**（與區塊對角遮罩等價，不必撐起整張 4307×4307 注意力矩陣）。本機 `gpt1.yaml` 仍可開 Flex Attention；Kaggle T4（sm_75）請關 Flex、走分層路徑。
- `n_layers` 在 VQ-VAE 的 Encoder / Decoder 裡是**注意力頭數**，不是層數；深度由 `enc_layers`（預設 0）與 `dec_layers`（預設 1）決定。
- 加入 gpt2 的 `<|pad|>` 後，pad id 通常是 50257（與官方 OWT 設定相同）。訓練與生成會以實際 tokenizer 為準，不硬假設這個數字。

## 授權

`models/`、`tokenization/`、`utils/` 與根目錄 `LICENSE` 來自 HazyResearch / hermann kumbong，MIT License。本專案新增的訓練與測試腳本同樣以 MIT 釋出。論文：Hermann Kumbong 等，*Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*，[arXiv:2610.14162](https://arxiv.org/abs/2610.14162)。
