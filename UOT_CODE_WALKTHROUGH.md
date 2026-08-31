# Đọc hiểu `models/uot.py` — giải thích từng phần & cách tinh chỉnh

File có đúng 2 thứ: **một hàm giải UOT** (`unbalanced_sinkhorn_log`) và **một
module fusion** (`UOTFusion`). Dưới đây đi từ toán → code → nút vặn.

---

## Phần 0 — Bài toán đang giải là gì

Ta có 16 vector mô tả frame video và 32 vector mô tả đoạn audio (đều 128 chiều).
Câu hỏi: **frame nào ứng với đoạn audio nào, và mạnh bao nhiêu?**

Optimal Transport trả lời bằng một **ma trận kế hoạch** `π ∈ R^{16×32}`, trong đó
`π[i,j]` = lượng "khối lượng" chuyển từ frame `i` sang audio slot `j`. Bài toán:

```
min_π  <C, π>  +  ε·KL(π ‖ a⊗b)  +  τ·KL(π1 ‖ a)  +  τ·KL(πᵀ1 ‖ b)
        └─┬──┘     └─────┬─────┘     └────────────┬────────────┘
      chi phí       entropic reg          ràng buộc biên (mềm)
      ghép cặp      → giải được          → phần "Unbalanced"
                      bằng Sinkhorn
```

Ba số hạng, ba vai trò:

| Số hạng | Vai trò | Nút vặn |
|---|---|---|
| `<C,π>` | ghép cặp nào rẻ thì ghép | thiết kế `C` |
| `ε·KL(π‖a⊗b)` | làm bài toán trơn & giải được lặp | `eps` |
| `τ·KL(π1‖a)` + `τ·KL(πᵀ1‖b)` | **cho phép** biên lệch khỏi đều | `tau` |

**Balanced OT** = ép cứng `π1 = a` và `πᵀ1 = b` (tương đương `τ = ∞`): mọi frame
*bắt buộc* nhận đúng 1/16 khối lượng, mọi audio slot *bắt buộc* được dùng.
**Unbalanced OT** thay ràng buộc cứng đó bằng phạt mềm KL với hệ số `τ`. Khi
`τ` hữu hạn, model được phép **vứt bỏ** audio slot vô nghĩa (im lặng, nhạc nền)
và **giảm khối lượng** ở frame không có bằng chứng audio. Đó là toàn bộ luận điểm
khoa học của việc dùng UOT thay vì OT thường ở đây.

---

## Phần 1 — `unbalanced_sinkhorn_log` (dòng 23–57)

### 1.1 Tại sao ở log-domain

Nghiệm có dạng `π = diag(u)·K·diag(v)` với `K = exp(-C/ε)`. Với `ε = 0.05` và
`C` cỡ 1 thì `exp(-1/0.05) = exp(-20) ≈ 2e-9` — nhân/chia mấy trăm lần ở fp16
sẽ underflow thành 0 và ra `NaN`. Nên ta làm việc với **thế vị (potential)**
`f = ε·log u`, `g = ε·log v` và dùng `logsumexp`. Không bao giờ tính `exp(-C/ε)` trực tiếp.

### 1.2 Vòng lặp (dòng 52–54)

```python
scale = tau / (tau + eps)                                    # dòng 50
for _ in range(n_iters):
    f = -scale * eps * logsumexp((g - C)/eps + log_b, dim=2)  # cập nhật biên hàng
    g = -scale * eps * logsumexp((f - C)/eps + log_a, dim=1)  # cập nhật biên cột
```

Đây là "scaling iteration" của Chizat/Séjourné. Hai dòng luân phiên nhau kéo biên
hàng và biên cột về gần `a`, `b`. **Toàn bộ khác biệt giữa OT thường và UOT nằm
ở hệ số `scale`:**

- `τ → ∞` ⇒ `scale → 1` ⇒ đúng thuật toán Sinkhorn cân bằng (biên khớp *chính xác*).
- `τ` nhỏ ⇒ `scale < 1` ⇒ mỗi bước chỉ *kéo một phần* về phía biên mục tiêu ⇒ biên
  được phép lệch ⇒ khối lượng bị mất/tạo ra.

Ví dụ `ε=0.05`: `τ=1.0 → scale=0.952`; `τ=0.1 → scale=0.667`.

### 1.3 Dựng lại `π` (dòng 56–57)

```python
log_pi = (f + g - C)/eps + log_a + log_b
return log_pi.exp()
```

`.exp()` chỉ gọi **một lần ở cuối**, khi số mũ đã được `f,g` cân bằng về gần 0 → an toàn.

### 1.4 Điểm quan trọng: **không** chuẩn hoá `π`

`π.sum()` **≤ 1** và ta cố ý giữ nguyên. Tổng khối lượng chính là tín hiệu
*"mẫu này audio và video khớp nhau tới mức nào"* — nếu chuẩn hoá về 1 thì vứt mất
đúng cái thông tin mà UOT tạo ra, và module tụt về thành cross-attention thường.

### 1.5 Số liệu đã đo (16 frame × 32 audio slot, cost = 1 − cosine)

Thí nghiệm 1 — cố ý làm 10/32 audio slot thành nhiễu ở xa mọi frame:

| | mass rơi vào 10 slot nhiễu |
|---|---|
| Balanced (`τ=∞`) | **0.3125** ← bị *ép* phải nhận |
| UOT (`τ=0.1`) | **0.0000** |

Thí nghiệm 2 — dữ liệu có alignment thật (audio slot `2i`, `2i+1` ứng frame `i`):

| eps | tau | tổng mass | hệ số scale mỗi hàng | độ tập trung top-1 | alignment đúng |
|---|---|---|---|---|---|
| 0.01 | ∞ | 1.000 | 1.00 | 0.500 | 100% |
| 0.05 | ∞ | 1.000 | 1.00 | 0.500 | 100% |
| 0.05 | 1.0 | 0.944 | 0.94–0.95 | 0.502 | 100% |
| 0.05 | 0.1 | 0.470 | 0.44–0.50 | 0.514 | 100% |
| **0.5** | ∞ | 1.000 | 1.00 | **0.155** ← nhoè | 100% |

(top-1 = 0.50 là *đúng*: mỗi frame khớp 2 audio slot, chia đôi 0.5/0.5.)

---

## Phần 2 — `UOTFusion` (dòng 60–146)

### 2.1 Lấy token: từ hàng trăm token xuống 16 và 32 (dòng 96–107)

```python
def _video_tokens(self, image_lowdim, n, t):
    x = image_lowdim.view(n, t, -1, self.dim)   # [n*16, 203, 128] -> [n, 16, 203, 128]
    return x[:, :, 0, :]                        # CLS mỗi frame -> [n, 16, 128]

def _audio_tokens(self, audio_lowdim):
    patches = audio_lowdim[:, 1:1+32*8, :]              # bỏ CLS & prompt -> [n, 256, 128]
    return patches.view(-1, 32, 8, 128).mean(dim=2)     # gộp theo tần số -> [n, 32, 128]
```

Vì sao `view(-1, 32, 8, 128)` là đúng: spectrogram `512×128`, `PatchEmbed_new`
dùng patch 16 stride 16 → lưới `32 (thời gian) × 8 (tần số)`, `flatten(2)` cho thứ tự
row-major nên token `k = t_idx*8 + f_idx`. Gộp `mean` theo trục tần số cho ra
**32 lát thời gian** — đúng thứ đối xứng với 16 frame video.

> ⚠ **Ràng buộc thứ tự (dễ sai âm thầm):** `Generate_Model.forward` làm
> `image.view(-1, c, h, w)` từ `[n, t, c, h, w]`, nên index của batch phẳng là
> `sample*16 + frame`. `_video_tokens` dùng `.view(n, t, ...)` và cuối hàm
> `a2v.reshape(n*t, 1, D)` — cùng một thứ tự. Nếu đổi thành `permute`/`transpose`
> ở một chỗ mà quên chỗ kia, model **vẫn chạy, vẫn hội tụ, chỉ là kém hơn** và
> rất khó phát hiện.

### 2.2 Ma trận chi phí (dòng 118–121)

```python
v_n = F.normalize(self.norm_v(v), dim=-1)      # LayerNorm rồi chuẩn hoá L2
a_n = F.normalize(self.norm_a(a), dim=-1)
C = 1.0 - torch.bmm(v_n, a_n.transpose(1, 2))  # [n, 16, 32], giá trị trong [0, 2]
```

`LayerNorm` riêng cho từng modality vì thang đo latent của video và audio khác nhau;
chuẩn hoá L2 để `C` là khoảng cách cosine, **giá trị nằm trong `[0,2]` bất kể block
nào**. Điều này quan trọng: `ε` là đại lượng *tương đối so với thang của `C`*, nếu
`C` co giãn theo block thì cùng một `ε` sẽ có tác dụng khác nhau ở mỗi tầng.

### 2.3 Dùng `π` để truyền tin (dòng 126–137)

```python
row_mass = pi.sum(dim=2, keepdim=True)                 # [n,16,1] frame i khớp được bao nhiêu
col_mass = pi.sum(dim=1, keepdim=True).transpose(1,2)  # [n,32,1] slot j được dùng bao nhiêu

a2v = bmm(pi / row_mass, a)   # HƯỚNG: audio -> video, [n,16,128]
v2a = bmm(piᵀ / col_mass, v)  # HƯỚNG: video -> audio, [n,32,128]

a2v = a2v * (row_mass * 16)   # nhân lại khối lượng tương đối
v2a = v2a * (col_mass * 32)
```

Tách làm **hai bước có chủ đích**:

1. `pi / row_mass` → mỗi hàng tổng bằng 1 = một **phân phối attention mềm**. Nhân
   với `a` cho ra: *"frame i nên nghe đoạn audio nào"*. Đây là phần **hướng**.
2. `* (row_mass * 16)` → nhân lại hệ số bằng 1.0 nếu frame nhận đúng phần đều
   (1/16), nhỏ hơn 1 nếu frame không khớp được gì. Đây là phần **cường độ**, và là
   chỗ duy nhất tính "unbalanced" thực sự đi vào feature.

Bỏ bước 2 đi thì module thoái hoá thành cross-attention có trọng số Sinkhorn —
vẫn chạy nhưng **mất đúng cái đang muốn kiểm chứng**.

`clamp_min(1e-8)` chống chia 0 khi `τ` rất nhỏ làm cả hàng về ~0.

### 2.4 Gate khởi tạo 0 (dòng 90–91, 139–140)

```python
self.gate_a2v = nn.Parameter(torch.zeros(1))
...
a2v = torch.tanh(self.gate_a2v) * self.proj_a2v(a2v)
```

`tanh(0) = 0` ⇒ **tại step 0, output của model trùng khít baseline**. Ý nghĩa:

- training không bị cú sốc từ một nhánh ngẫu nhiên chưa học gì,
- mọi thay đổi UAR/WAR đo được đều quy được về UOT,
- cho phép warm-start từ checkpoint baseline (xem `UOT_INTEGRATION.md` §4).

Đây cùng ý tưởng với gate của chính MMA-DFER (`self.all_gate` khởi tạo `zeros`),
và với zero-init gate của Flamingo / LoRA.

> **Hệ quả sẽ thấy trong smoke test:** ở step đầu tiên, gradient của `proj_*` và
> `norm_*` **bằng đúng 0** (vì chúng nằm sau `tanh(gate)=0`), chỉ có gradient của
> **gate** là khác 0. Đó là *bình thường*, không phải lỗi. `tools/smoke_test.py`
> đã in tách hai nhóm và ghi rõ nhóm nào phải khác 0.

### 2.5 Trả về (dòng 142–146)

```python
a2v = a2v.reshape(n * t, 1, self.dim)   # [n*16, 1, 128] -> broadcast lên 203 token ảnh
v2a = v2a.mean(dim=1, keepdim=True)     # [n, 1, 128]    -> broadcast lên 263 token audio
```

Hình dạng `[.., 1, 128]` là để cộng thẳng vào `image_lowdim_norm2` / `audio_lowdim2`
nhờ broadcasting, khớp y hệt cách baseline cộng vector mean-pool.

> ⚠ **Đây là chỗ thoả hiệp lớn nhất của bản hiện tại**, và là ứng viên số 1 để
> tinh chỉnh: chiều **audio→video có alignment thật** (mỗi frame nhận vector riêng
> của nó), nhưng chiều **video→audio thì `.mean(dim=1)` bóp 32 lát thời gian
> thành 1 vector duy nhất** — tức là vứt bỏ alignment vừa tính được. Cách sửa ở §3.1.

---

## Phần 3 — Các nút vặn, theo thứ tự đáng thử

### 3.1 [Ưu tiên cao] Trả `v2a` theo từng token audio thay vì mean

Sửa 2 dòng cuối `UOTFusion.forward`:

```python
        a2v = a2v.reshape(n * t, 1, self.dim)

        # v2a: [n, 32, D] -> [n, 263, D] khớp đúng layout token của audio stream
        v2a_mean = v2a.mean(dim=1, keepdim=True)                       # cho CLS + prompt
        v2a_patch = v2a.repeat_interleave(self.n_audio_f, dim=1)       # 32 -> 256 (trải theo tần số)
        n_prompt = audio_lowdim.shape[1] - 1 - self.n_audio_t * self.n_audio_f
        v2a = torch.cat([v2a_mean,
                         v2a_patch,
                         v2a_mean.expand(-1, n_prompt, -1)], dim=1)    # [n, 263, D]
        return a2v, v2a
```

Không cần đổi gì ở `Generate_Model.py` (`audio_lowdim2 + v2a` vẫn cộng đúng vì
shape khớp chính xác `[n, 263, 128]` thay vì broadcast). Đây là cách dùng `π` đầy đủ.

### 3.2 `tau` — nút quan trọng nhất, chính là biến độc lập của thí nghiệm

| `tau` | Ý nghĩa | Dùng khi |
|---|---|---|
| `1e6` | ≈ Balanced OT | **control bắt buộc** — chứng minh "U" trong UOT có tác dụng |
| `1.0` | relaxation nhẹ (mass ~0.94) | mặc định, an toàn |
| `0.1` | relaxation mạnh (mass ~0.47) | khi data nhiều audio nhiễu (MAFW) |
| `0.01` | gần như vứt hết | quá mạnh, mass ~0.003 — chỉ để thấy giới hạn |

Ablation tối thiểu để bảo vệ kết quả: `{baseline, τ=1e6, τ=1.0, τ=0.1}` trên fold 1.
Nếu `τ=1e6` ≈ `τ=1.0` thì tính "unbalanced" không đóng góp gì và phải nói thẳng.

### 3.3 `eps` — độ sắc của plan

`C ∈ [0,2]` nên `eps` cần cùng thang đó. Đã đo: `eps=0.5` làm plan nhoè hẳn
(top-1 tụt 0.50 → 0.155). Vùng nên dùng: **`0.01 – 0.1`**, mặc định `0.05`.
`eps` quá nhỏ (<0.01) → plan gần one-hot, gradient thưa và dễ mất ổn định.

### 3.4 `n_iters` — đánh đổi chi phí/độ chính xác

Đã đo hội tụ (`eps=0.05, τ=1.0`): mass = 0.991 (1 vòng) → 0.944 (10) → 0.912 (200).
Tức **10 vòng chưa hội tụ hẳn**. Điều đó chấp nhận được vì đây là *một lớp mạng có
độ sâu cố định, khả vi*, không phải solver chính xác — miễn là train và test dùng
**cùng** `n_iters`. Muốn trung thành với UOT thật thì để `n_iters=50` kèm
`--uot-detach` (xem 3.5) để khỏi phải backprop qua 50 vòng.

### 3.5 `detach_plan`

`False` (mặc định): gradient chảy ngược qua toàn bộ `n_iters` vòng lặp về tới
`norm_v/norm_a` — model **học cách tạo ra không gian sao cho OT ghép cặp tốt**.
Đắt hơn nhưng mạnh hơn.
`True`: coi `π` như trọng số routing cố định. Rẻ, ổn định, cho phép tăng `n_iters`.
Đáng thử nếu training dao động.

### 3.6 `video_token`

`'cls'` (mặc định) — CLS token của mỗi frame. Đổi thành bất kỳ giá trị nào khác
(vd `'mean'`) thì dùng trung bình 196 patch token. CLS thường mang thông tin toàn
cục tốt hơn; `'mean'` ổn định hơn với nhiễu. Rẻ để thử.

### 3.7 Chia sẻ tham số giữa các block

Hiện tại mỗi block có một `UOTFusion` riêng ⇒ ~0.4M tham số. Nếu overfit (MAFW
nhỏ, có lớp chỉ ~145 mẫu), thử dùng **một** instance dùng chung cho cả 12 block, hoặc
chỉ gắn UOT ở các block sau (vd `ii >= 6`) nơi feature đã có ngữ nghĩa:

```python
if self.use_uot and ii >= 6:
    ...
```

### 3.8 Cho `eps`/`tau` học được

Đang là số thường (`self.eps`, `self.tau`), không phải `nn.Parameter`. Muốn học:
tham số hoá ở log-domain để luôn dương —
`self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))` rồi dùng
`self.log_tau.exp()`. Cẩn thận: `τ` có xu hướng bị đẩy lên rất lớn (quay về
balanced) vì đó là hướng dễ giảm loss ngắn hạn. Nên thử sau khi đã có kết quả
với `τ` cố định.

---

## Phần 4 — Chi phí tính toán

Mỗi block: 2 `logsumexp` trên tensor `[n, 16, 32]` × 10 vòng. Với `n=8` là
~82k phần tử/vòng — không đáng kể so với một block ViT-B trên 128 ảnh.
Bộ nhớ: unrolled 10 vòng lưu ~20 tensor `[8,16,32]` ≈ vài trăm KB.
Tham số thêm: `2×(128×128+128) + 2×2×128 + 2 ≈ 33.5k`/block × 12 ≈ **0.40M**.

Nói cách khác: **UOT gần như miễn phí ở đây**, chi phí thật nằm ở việc phải train lại.

---

## Phần 5 — Bản đồ nhanh

| Muốn đổi gì | Sửa ở đâu |
|---|---|
| balanced ↔ unbalanced | `tau` (CLI `--uot-tau`) |
| plan sắc/nhoè | `eps` (CLI `--uot-eps`) |
| độ chính xác solver | `n_iters` (CLI `--uot-iters`) |
| có backprop qua solver không | `detach_plan` (CLI `--uot-detach`) |
| định nghĩa "giống nhau" giữa 2 modality | `C = 1 - cosine`, dòng 121 |
| video lấy token nào | `_video_tokens`, dòng 96–101 |
| audio gộp thế nào | `_audio_tokens`, dòng 103–107 |
| dùng mass hay bỏ mass | dòng 136–137 |
| UOT mạnh/yếu lúc bắt đầu | init của `gate_*`, dòng 90–91 |
| UOT cộng thêm hay thay thế mean-pool | `Generate_Model.forward` (xem `UOT_INTEGRATION.md` §3.2) |
