# KVarN Q4K：EXL3 路徑的 4-bit KV cache

| 項目 | 內容 |
|---|---|
| 狀態 | 已實作（v0.7.8 之後，未釋出） |
| 目標 repo | `sf-stav/veloGB10` |
| 基準版本 | `main` @ `438176b`（v0.7.8） |
| 撰寫日期 | 2026-11-02 |
| 目標格式 | `--kv-cache q4k`（K 與 V 都是 4-bit，H32 旋轉） |
| 主要目標 | 在品質對齊 q8 的前提下，把 KV cache 容量提高約 1.9 倍 |

## 1. 摘要

在 EXL3 推論路徑新增 `q4k`：K 與 V 都用 4-bit，沿用 q8 的 H32 旋轉與 per-32-group
f16 scale，兩個 dim 併成一個 byte。hd=256 時一個 row 是 144 B（q8 是 272 B）。

實測（本地 Flash-Next 3.05 pack，8 windows × 2,048 tokens）：

| text | q8 | q4k | 差距 |
|---|---|---|---|
| code | 3.9768 | 3.9792 | +0.06% |
| prose | 6.8758 | 6.8727 | −0.05% |
| json | 1.4098 | 1.4098 | 0.00% |

262K context、4 lanes：trunk KV 從 13.69 GB 降到 7.25 GB（1.89×），每 lane 3.76 GB，
可容 **8 lanes**（q8 是 5）。吞吐與 q8 同級（1/2/4 streams：63.7 / 84.7 / 92.9 對
66.0 / 88.3 / 92.3 tok/s）。

## 2. 前因後果

### 2.1 起點

這個工作源自一輪 4-bit 與 sub-4-bit KV cache 量化的調研（KVarN、TurboQuant、OCTOPUS、
HyperQuant、NOVA-KV、JoLT、TaSQ、KVQuant、KIVI、PolarQuant、QuaRot、OSCAR）。結論：

1. 4-bit 已經接近無損，差距主要在 2–3 bit。
2. 多數 sub-4-bit 方法只省容量，不省時間（decode kernel 慢 5–66 倍）。
3. KVarN 是少數同時贏精度與吞吐的方案：H32 旋轉壓離群值、RTN 讓 dequant 便宜。

### 2.2 為什麼選 4-bit（而不是 K4/V2）

KVarN 出貨的 `kvarn_k4v2_g128` 是 K4/V2。第一版照著做了 K4/V2，量測結果：

| text | q8 | K4/V2 | 差距 |
|---|---|---|---|
| code | 3.9768 | 4.0755 | +2.48% |
| prose | 6.8758 | 7.0811 | +2.99% |
| json | 1.4098 | 1.4131 | +0.23% |

2-bit V 的誤差超出 0.5% 門檻，而且不是 scale 問題：absmax 校準（×0.625）與 RMS-based
scale 都試過，PPL 反而更差（8.31 / 8.06 對 8.06）。原因是 4 個 code level 對 Gaussian
本來就不夠，KVarN 靠 128-token tile 的 variance normalization（Sinkhorn）補償，而我們
沒有做（理由見 D2）。

同一組實驗把 V 換成 4-bit（K4/V4）後，prose PPL 是 7.7938，**比 q8 的 7.8175 還好**。
所以 2-bit V 是唯一的品質瓶頸，4-bit V 是品質安全點。最終只保留 4-bit。

### 2.3 為什麼不是 NVFP4

NVFP4（E2M1 + E4M3 scale）在 GB10 的優勢是 tensor core 原生 FP4 mma；但 EXL3 的
attention kernel 是 scalar warp path（長 context decode 是 byte-bound），用不到
tensor core。頻寬上 FP4 與 int4 都是 0.5 B/elem，codebook 品質差距又被 H32 稀釋。
NVFP4 的戰場是權重 GEMM 與 indexer scoring，不是 EXL3 的 KV cache。

## 3. 設計決策

### D1：沿用 H32，不用 full-dim Hadamard

scale 是 per-32-group absmax。full-dim 旋轉會把不同 group 的 channel 混在一起，per-group
scale 就失去意義。H32 與 per-32-group scale 是配套的。q8 已經這樣做且品質通過。

### D2：不做 128-token tile Sinkhorn

- 4 bps 時各 bias-correction 變體差距在 0.04 PPL 內（HyperQuant）。
- Sinkhorn 需要 tile staging、commit、rollback，會撞上 byte-identical 與 CUDA graph 約束。
- 4-bit 不需要它：實測已經在 q8 的噪音內。

### D3：K 與 V 對稱，單一 `rb` / `fmt`

只支援一種 4-bit 格式，K 與 V 的 row 大小相同，所以 `XqKv` 保持 `{ k, v, rb, fmt, hd }`。
這讓 q8/fp8/f16 的既有程式碼（38 處 `kc.rb`、spill、prefix、draft 診斷）完全不用改。

### D4：不做 sink/tail FP16 保護

q8 沒有做且品質通過；4-bit 的實測 PPL 也在噪音內，第一版不做。

### D5：單一格式常數

`KV_Q4K = 5`，CLI 名稱 `q4k`（`q4` 已被 NVFP4 路徑使用）。

## 4. 格式規格

### 4.1 Row 佈局（hd = head_dim）

```
offset 0            : hd/2 bytes   packed 4-bit codes，2 個 dim 一個 byte
offset hd/2         : hd/32 * 2 B  f16 group scales，每 32 個 dim 一個
offset hd/2 + hd/16 : padding 到 16 B 對齊
```

dim `d` 在 byte `d/2` 的第 `(d % 2) * 4` 個 bit，低位在前（偶數 dim 在低 nibble）。

### 4.2 大小

| hd | q8 row | q4k row | 倍率 |
|---|---|---|---|
| 128 | 144 B | 80 B | 1.80× |
| 256 | 272 B | 144 B | 1.89× |

bpe：hd=256 時 q8 = 8.5、q4k = 4.5。

### 4.3 量化公式

旋轉與 scale 沿用 q8：

```
v = H32(x) * (1/sqrt(32))
s = max over 32-group |v| + 1e-10     // f16
code = clamp(floor(v/s * 8 + 8), 0, 15)
dequant = (code - 7.5) * (s / 8)
```

### 4.4 與 q4x 的關係

`q4x` 是 WP25 的 probe-only 對照組：同樣的 H32 + per-32 scale + 4-bit 中點網格，但把
4-bit code 塞進 q8 的 byte（`q4*16+8`），row 仍是 272 B。q4k 是「同一個量化器裝進一半的
row」，並修正了 q4x 的 `s/256` 固定偏移。合成列 SNR：q4k 22.98 dB、q4x 22.93 dB、q8
47.07 dB（每少 2 bit 約掉 12 dB，符合理論）。

## 5. 實作

| Commit | 內容 |
|---|---|
| `q4k: format constant, row bytes and the host reference` | `KV_Q4K`、`kv_rowbytes`、`q4k_quant_row`/`q4k_dequant_row`、probe host 段、4 個單元測試 |
| `q4k: write and read the packed 4-bit row` | `xq_kv_rowb`/`put`/`ld`/`ld8`、probe device 段（bitwise） |
| `q4k: keep flash256 and gate the single-FMT fast paths` | flash256 的 q4k case；dense-v3 以外仍以 q8 佈局為前提的快速路徑對 q4k 關閉 |
| `q4k: dense-v3 stages and dequantizes the packed 4-bit row` | `xq_dv3_stage_v` 的 scale offset、`xq_dv3_ldv_q4k`、`P4` constexpr、四個 acc entry、binv probe |
| `q4k: serve, lane fit and lane packing` | serve 解析、lane fit 0.5625 B/elem、packing gate、CHANGELOG |

關鍵細節：

- `xq_kv_put_q8` 用 `__shfl_xor_sync` 讓兩個 lane 的 code 併成一個 byte，scale 寫在
  `row + hd/2`。
- `xq_kv_ld8` 一次讀 4 B（8 個 nibble），`xq_kv_ld` 讀單一 nibble。
- dense-v3：`N16 = SD/32`、`RS = SD/2`、`d0b = blockIdx.x * SD/2`，scale chunk 在
  `src + hd/2`；輸出 un-rotation 直接重用 q8 的分支。
- flash256：`a3fg_stage_v<XQ_KV_Q4K, ...>`，prefill 維持快路徑。

## 6. 驗證結果

### 6.1 Bitwise

- `--probe-exl3-kvq --kv-cache q4k`：device 與 host reference 的 bytes / deq / rec 全部
  0 mismatch；`ld8 == ld`；零列精確為 0。
- `--exl3-dense-xcheck=1`（單 slot 的 verify 與 decode step）：0 mismatching halves。
- `--probe-exl3-pack2` / `--probe-exl3-packstate`：q8 與 q4k 都 PASS
  （tokens/accepts 相同、`snap_diff 0`）。

### 6.2 品質

見 §1 的 PPL 表。`--exl3-dense-v3=1` 與 `=0` 的 PPL 完全相同（PPL 走 prefill，不經
dense-v3；這也確認兩條 decode 路徑沒有分歧）。

### 6.3 容量（262K，實測）

| 格式 | per-lane KV | 262K lanes | trunk KV @262K |
|---|---|---|---|
| q8 | 5.61 GB | 5 | 13.69 GB |
| q4k | 3.76 GB | **8** | 7.25 GB |

### 6.4 吞吐（8K、`--prefix-cache off`、128 output tokens、best of 3）

| 並發 | q8 | q4k |
|---|---|---|
| 1 | 66.0 | 63.7 |
| 2 | 88.3 | 84.7 |
| 4 | 92.3 | 92.9 |

4-stream 的上限是權重頻寬，與 KV 格式無關。1/2-stream 的 2–4% 差距來自 4-bit unpack
與部分快速路徑（hg/union/pq8/W4S）仍只支援 q8。

## 7. 已知限制（刻意保留）

`xq_attn_decode_group`（v2 與 loop 兩個 body）假設**整個 group 只有一個 slot**：

```c
const int slot = slotpos[0 * 2 + 0];  // one slot for the whole group
```

單一請求的 verify 成立；packed verify（兩個請求、兩個 slot）不成立，segment B 的 K/V
會被讀寫到 segment A 的 slot。q8 有一樣的潛伏 bug，只是 q8 的 packed verify 預設走
dense-v3（逐 row slot），所以踩不到。q4k 也走 dense-v3，因此同樣踩不到。

**這不是本次修改造成的，也沒有修**：決定保留原狀，因為：

- 出貨設定（Flash-Next + 預設 tune + `--lane-pack on`）一定走 dense-v3。
- 修它會動到 q8 共用的 kernel；保留則 q8 的程式碼完全不變。

**使用限制**：`--lane-pack on` 不可與 `--exl3-dense-v3=0` 同時使用（q8 與 q4k 皆然）。
`--exl3-dense-xcheck=1` 在 packed verify 上會把 segment B 判為 mismatch，這是參考
kernel 的預期行為，不是 dense-v3 的錯。

**待辦**：向上游回報這個假設是否有意為之（是否要在 `xq_attn_decode_group` 改成逐 row
slot，或是在 host 端拒絕「packing + dense-v3 off」的組合）。

## 8. 尚未支援

- **2-bit V**：品質不足（§2.2），除非做 tile Sinkhorn 或 token scale，不在目前範圍。
- **3-bit V**：packing 非對齊，工程量高，未做。
- **q4k 的 hg1/hg3、union、pq8、W4S dattn**：這些路徑以 q8 佈局做 16 B chunk 複製或
  單一 compile-time FMT staging，對 q4k 關閉。QSA 走 `xq_attn_sel_splitk4`，decode 走
  dense-v3 或 `xq_attn_decode`。
- **indexer planes 量化**：仍是 bf16 獨立 plane（約 2.2 GB），是容量的下一個瓶頸。
- **TP=2/4 的硬體實測**：只驗證了 fmt 傳遞與握手邏輯。

## 9. 後續工作

1. 把 hg/union/pq8/W4S 也接上 q4k（需要把 16 B chunk 複製改成 packed row）。
2. indexer keys 量化（FP8 最簡單，約 2×）。
3. 2-bit V 的品質補強（token scale 先試，再 Sinkhorn）。
4. 上游回報 §7 的假設。

## 10. 變更紀錄

| 日期 | 內容 |
|---|---|
| 2026-11-02 | 初稿。基準 v0.7.8，`main` @ `438176b`；q4k 實作完成並量測。 |
