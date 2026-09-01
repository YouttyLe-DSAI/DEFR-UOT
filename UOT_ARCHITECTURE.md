# Kiến trúc UOT trong MMA-DFER

Xem thêm: `UOT_DIFF.md` (UOT đụng vào những phần nào của pipeline gốc) và
`UOT_INTEGRATION.md` (kế hoạch tích hợp, ngân sách train lại).

---

## 1. Vị trí trong pipeline

```
video 16 frame ─→ MAE-Face ViT-B (ĐÓNG BĂNG) ─┐
                                               │   vòng lặp 12 block
audio spectrogram ─→ AudioMAE ViT-B (ĐÓNG BĂNG)┘
                                               ▼
     ┌──────────────── TRONG MỖI BLOCK ii = 0..11 ────────────────┐
     │  image [n·16, 203, 768] ──temporal_pre───→   [.., 128]     │
     │  audio [n,    263, 768] ──audio_proj_pre─→   [.., 128]     │
     │                                                            │
     │  ① fusion GỐC : mean-pool broadcast      (GIỮ NGUYÊN)      │
     │  ② UOT        : a2v, v2a                 (CỘNG THÊM)       │
     │                                                            │
     │  forward_block_post ──→ x + tanh(all_gate) · x_t           │
     └────────────────────────────────────────────────────────────┘
                                               ▼
                    temporal_net → our_classifier → C lớp
```

UOT chạy **12 lần mỗi forward**, mỗi block một instance `UOTFusion` riêng.

Hai encoder **luôn đóng băng** — đó là thiết kế của MMA-DFER, không liên quan tới UOT.
Chỉ adapter, classifier, temporal_net và uot_fusion được train.

## 2. Điểm chèn — input / output

```python
image_lowdim_norm2 = image_lowdim_norm + audio_lowdim.mean(1)...      # fusion gốc
audio_lowdim2      = audio_lowdim + image_lowdim_norm.view(...)...    # fusion gốc

if self.use_uot:                                                      # <-- UOT
    a2v, v2a = self.uot_fusion[ii](image_lowdim_norm, audio_lowdim, n, t)
    image_lowdim_norm2 = image_lowdim_norm2 + a2v
    audio_lowdim2      = audio_lowdim2 + v2a
```

| | Shape | Ghi chú |
|---|---|---|
| vào `image_lowdim_norm` | `[n·16, 203, 128]` | 203 = 1 CLS + 196 patch + 6 prompt |
| vào `audio_lowdim` | `[n, 263, 128]` | 263 = 1 CLS + 256 patch + 6 prompt |
| ra `a2v` | `[n·16, 1, 128]` | broadcast lên 203 token ảnh |
| ra `v2a` | `[n, 1, 128]` | broadcast lên 263 token audio |

**Hai dòng fusion gốc được giữ nguyên.** UOT chỉ cộng thêm qua một gate khởi tạo 0, nên
tắt cờ `--use-uot` là về đúng baseline, không sai một bit.

> ⚠ Thứ tự batch phẳng là `sample*16 + frame` (do `image.view(-1, c, h, w)`).
> `_video_tokens` dùng `.view(n, t, ...)` và cuối hàm `a2v.reshape(n*t, 1, D)` — cùng một
> thứ tự. Đổi sang `permute` ở một chỗ mà quên chỗ kia thì model **vẫn chạy, vẫn hội tụ,
> chỉ kém hơn** và rất khó phát hiện.

## 3. Bên trong `UOTFusion` — 5 bước

### ① Rút gọn token về trục thời gian

```python
v = image_lowdim.view(n, 16, 203, 128)[:, :, 0, :]        # [n, 16, 128]  CLS mỗi frame
a = audio_lowdim[:, 1:257].view(n, 32, 8, 128).mean(2)    # [n, 32, 128]  gộp theo tần số
```

`32 × 8` vì spectrogram `512×128`, patch 16 stride 16 → lưới **32 thời gian × 8 tần số**,
flatten row-major nên token `k = t_idx*8 + f_idx`. Gộp `mean` theo tần số cho ra 32 lát
thời gian — đối xứng với 16 frame video.

### ② Cost matrix

```python
v_n = F.normalize(norm_v(v), dim=-1)
a_n = F.normalize(norm_a(a), dim=-1)
C   = 1.0 - v_n @ a_n.transpose(1, 2)        # [n, 16, 32], giá trị trong [0, 2]
```

Bài toán OT chỉ **16 × 32 mỗi mẫu**. LayerNorm riêng từng modality vì thang latent khác
nhau; chuẩn hoá L2 để `C` luôn nằm trong `[0,2]` bất kể block nào — quan trọng vì `ε` là
đại lượng *tương đối so với thang của `C`*.

### ③ Giải UOT

```
min_π  ⟨C,π⟩ + ε·KL(π‖a⊗b) + τ·KL(π1‖a) + τ·KL(πᵀ1‖b)
       └─┬─┘   └────┬────┘   └──────────┬──────────┘
     chi phí    làm trơn      ràng buộc biên MỀM = phần "Unbalanced"
```

Balanced OT ép cứng `π1 = a` và `πᵀ1 = b`. UOT thay bằng phạt KL hệ số `τ`, cho phép
tạo/huỷ khối lượng. Toàn bộ khác biệt nằm ở **một biến**:

```python
scale = tau / (tau + eps)        # τ→∞ ⇒ scale=1 ⇒ Sinkhorn cân bằng
f = -scale * eps * logsumexp((g - C)/eps + log_b, dim=2)
g = -scale * eps * logsumexp((f - C)/eps + log_a, dim=1)
```

Làm ở log-domain vì `exp(-C/ε)` với `ε=0.05` cho ra `~2e-9`, fp16 sẽ underflow thành `NaN`.

### ④ Dùng plan — tách hướng và cường độ

```python
row_mass = π.sum(2)                    # [n,16,1] frame i khớp được bao nhiêu
a2v = (π / row_mass) @ a               # HƯỚNG   : frame i nên nghe đoạn nào
a2v = a2v * (row_mass * 16)            # CƯỜNG ĐỘ: đây là phần "unbalanced"
```

Bước 2 là **chỗ duy nhất** tính unbalanced đi vào feature. Bỏ nó thì module tụt xuống
thành cross-attention có trọng số Sinkhorn — vẫn chạy, nhưng mất đúng thứ cần kiểm chứng.

### ⑤ Gate khởi tạo 0

```python
a2v = torch.tanh(self.gate_a2v) * self.proj_a2v(a2v)     # tanh(0) = 0
```

Ở step 0, output bằng đúng 0 ⇒ model trùng khít baseline. Cùng ý tưởng với `all_gate` của
chính MMA-DFER và với zero-init gate của Flamingo / LoRA.

**Hệ quả:** gradient của `proj_*` và `norm_*` bằng **đúng 0** ở step đầu (chúng nằm sau
`tanh(gate)=0`), và gradient của chính `gate_*` cũng bằng 0 vì `all_gate` của baseline
cũng zero-init. Đo được: `0` ở step 0 → `3.1e-07` ở step 1 → `8.2e-07` ở step 3. Đó là
khởi động chậm theo thiết kế, không phải đứt kết nối.

## 4. Code nằm ở đâu

| File | Sửa gì | Dòng |
|---|---|---|
| `models/uot.py` | file mới: solver + `UOTFusion` | +146 |
| `models/Generate_Model.py` | import, `ModuleList` trong `__init__`, 4 dòng trong `forward` | +26 |
| `main.py` | 5 cờ argparse, `requires_grad`, snapshot list | ~10 dòng UOT |
| `evaluate.py` | 5 cờ argparse | ~9 dòng UOT |

**7 file gốc không bị đụng một ký tự:** hai dataloader, `models_vit.py`,
`Temporal_Model.py`, `audio_models_vit.py`, `train_MAFW.sh`, `train_DFEW.sh`.

## 5. Mục đích

| | Fusion gốc | UOT |
|---|---|---|
| Cách làm | mean-pool audio → cộng vào **mọi** frame như nhau | mỗi frame nhận audio **của riêng nó** |
| Cấu trúc thời gian | vứt hết | giữ |
| Audio vô nghĩa (im lặng, nhạc nền) | vẫn bị trộn vào | được phép **bỏ** |

Dư địa có thật: baseline của nhóm đo `av − visual_only` = **+5 đến +10 UAR**. Nhánh audio
đóng góp lớn nhưng đang được dùng ở dạng thô nhất.

Vì sao OT chứ không phải attention: attention chuẩn hoá theo **hàng**, mỗi frame độc lập,
nên một audio slot có thể bị mọi frame giành lấy. OT ràng buộc **cả hai chiều** → phân bổ
nhất quán toàn cục. UOT thêm: slot không khớp gì được phép không nhận khối lượng.

## 6. Tinh chỉnh

### Nút vặn qua CLI

| Nút | Vùng nên dùng | Tác dụng |
|---|---|---|
| `--uot-tau` | `1e6` / `1.0` / `0.1` | **biến chính**. `1e6` ≈ balanced OT — ablation bắt buộc |
| `--uot-eps` | `0.01 – 0.1` | độ sắc của plan |
| `--uot-iters` | `10 – 50` | độ chính xác solver |
| `--uot-detach` | | tắt backprop qua solver: rẻ hơn, ổn định hơn |

Số đo trên dữ liệu tổng hợp (16 frame × 32 audio slot, cost = 1 − cosine):

| eps | tau | tổng mass | độ tập trung top-1 |
|---|---|---|---|
| 0.05 | ∞ (balanced) | 1.000 | 0.500 |
| 0.05 | 1.0 | 0.944 | 0.502 |
| 0.05 | 0.1 | 0.470 | 0.514 |
| **0.5** | ∞ | 1.000 | **0.155** ← nhoè hẳn |

`eps = 0.5` phá alignment. Giữ trong `0.01 – 0.1`.

Về `--uot-iters`: đo hội tụ (`eps=0.05, τ=1.0`) cho mass = 0.991 (1 vòng) → 0.944 (10) →
0.912 (200). **10 vòng chưa hội tụ hẳn.** Chấp nhận được vì đây là *một lớp mạng độ sâu
cố định, khả vi*, không phải solver chính xác — miễn train và test dùng **cùng** `n_iters`.
Muốn trung thành hơn thì `--uot-iters 50` kèm `--uot-detach`.

### Sửa trong code

| Muốn đổi | Ở đâu |
|---|---|
| định nghĩa "giống nhau" giữa 2 modality | `C = 1 - cosine`, `uot.py:121` |
| video lấy token nào | `_video_tokens`, dòng 96–101 |
| audio gộp thế nào | `_audio_tokens`, dòng 103–107 |
| dùng mass hay bỏ mass | dòng 136–137 |
| UOT mạnh/yếu lúc bắt đầu | init của `gate_*` |
| UOT cộng thêm hay **thay thế** mean-pool | `Generate_Model.forward` |

### Ablation tối thiểu

`{baseline, τ=1e6, τ=1.0, τ=0.1}` trên cùng fold. Nếu `τ=1e6` ≈ `τ=1.0` thì tính
"unbalanced" không đóng góp gì — và phải nói thẳng điều đó.

## 7. Điều cần biết trước khi tinh chỉnh

Hiệu quả của **mọi nút ở mục 6 đều phụ thuộc vào bước ②**, và bước ② đang là chỗ yếu nhất.

```python
temporal_pre[ii]   = nn.Linear(768, 128)              # phép chiếu A, học riêng
audio_proj_pre[ii] = nn.Linear(768, 128) + LayerNorm  # phép chiếu B, học riêng
C = 1 - cosine(norm_v(v), norm_a(a))
```

Không có ràng buộc nào buộc hai phép chiếu đưa cùng một khái niệm về cùng một vùng. Gốc
rễ sâu hơn: **MAE-Face và AudioMAE train hoàn toàn độc lập**, chưa bao giờ nhìn thấy nhau.

Nhóm đã đo: OT khôi phục tương ứng audio↔visual ở **6,5%** so với đoán mò **6,0%**.
Không phải OT dở — cost matrix đang là nhiễu, và OT giải tối ưu một bài toán vô nghĩa.

Bản hiện tại có cho gradient chảy ngược qua Sinkhorn về `norm_v`/`norm_a`, nên model *có
thể* tự học căn chỉnh. Nhưng tín hiệu rất yếu: mục tiêu gián tiếp (cross-entropy 7 lớp),
~6.000 clip, và đường gradient đi qua 2 lớp gate zero-init.

**Chỉnh `tau`/`eps` lúc này giống hiệu chỉnh một cái cân chưa được zero.** Việc đáng làm
trước là thêm một loss tương phản phụ ngay trong không gian 128-d:

```
z_v = mean(temporal_pre(image))        mỗi clip → 1 vector 128-d
z_a = mean(audio_proj_pre(audio))      mỗi clip → 1 vector 128-d
L_total = L_CE + λ · InfoNCE(z_v, z_a)     # clip i khớp audio i, không khớp audio j≠i
```

Nó **trực tiếp** buộc hai phép chiếu về cùng một không gian, sau đó `C[i,j]` mới đo được
thứ gì thật. ~20 dòng, không thêm tham số, không thêm bước forward. Hạn chế: batch 4–8
chỉ cho 3–7 negative mỗi mẫu.

## 8. Chi phí tính toán

Mỗi block: 2 `logsumexp` trên `[n, 16, 32]` × 10 vòng. Với `n=8` là ~82k phần tử/vòng —
không đáng kể so với một block ViT-B trên 128 ảnh.

Tham số thêm: `2×(128×128+128) + 2×2×128 + 2 ≈ 33.5k` mỗi block × 12 ≈ **0.40M**.

**UOT gần như miễn phí.** Chi phí thật nằm ở chỗ phải train lại — xem `UOT_INTEGRATION.md`.
