# Tích hợp UOT vào MMA-DFER — phân tích & kế hoạch code

## 1. Điểm fusion duy nhất của pipeline nằm ở đâu

Toàn bộ tương tác audio–video của MMA-DFER nằm trong **1 vòng lặp 12 block** ở
`models/Generate_Model.py`, hàm `GenerateModel.forward()` (dòng ~158–185):

```python
for ii in range(len(self.audio_model.blocks)):          # 12 block
    audio = self.audio_model.forward_block_pre(ii, audio)      # [n, 263, 768]
    image = self.image_encoder.forward_block_pre(ii, image, B) # [n*16, 203, 768]

    image_lowdim_temp = self.image_encoder.temporal_pre[ii](image)        # -> 128-d
    image_lowdim_norm = self.image_encoder.temporal_pre_norm[ii](image_lowdim_temp)
    audio_lowdim      = self.image_encoder.audio_proj_pre[ii](audio)      # -> 128-d

    # >>>>>>>>>>>>>>  TOÀN BỘ FUSION CHỈ CÓ 2 DÒNG NÀY  <<<<<<<<<<<<<<
    image_lowdim_norm2 = image_lowdim_norm + audio_lowdim.mean(1).unsqueeze(1).repeat_interleave(t,0)
    audio_lowdim2      = audio_lowdim + image_lowdim_norm.view(B//t, t, self.n_image+6+1, 128).mean(1).mean(1).unsqueeze(1)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    image = self.image_encoder.forward_block_post(ii, image, image_lowdim_norm2, B)
    audio = self.audio_model.forward_block_post(ii, audio, audio_lowdim2)
```

Ý nghĩa 2 dòng đó: **mean-pool toàn bộ token của một modality thành 1 vector rồi
cộng như một bias toàn cục vào mọi token của modality kia.** Không hề có
alignment ở mức token — mọi khung hình được coi là liên quan như nhau tới mọi
đoạn audio.

Đây chính xác là chỗ UOT phải thay thế, và là **chỗ duy nhất** cần đụng vào.
(Còn một điểm fusion nhỏ nữa ở dòng 181, `self.vision_proj(image + audio.unsqueeze(1))`,
nhưng nó chỉ cộng CLS token cuối cùng — không phải nơi đặt UOT.)

### Shape thực tế tại điểm fusion (img-size 224, 16 frame)

| Tensor | Shape | Ghi chú |
|---|---|---|
| `image_lowdim_norm` | `[n*16, 203, 128]` | 203 = 1 CLS + 196 patch + 6 prompt |
| `audio_lowdim` | `[n, 263, 128]` | 263 = 1 CLS + 256 patch + 6 prompt |

Layout của 256 audio patch: spectrogram `512×128`, patch 16 stride 16
→ lưới **32 (thời gian) × 8 (tần số)**, flatten row-major. Nên
`audio_lowdim[:, 1:257].view(n, 32, 8, 128).mean(2)` cho ra **32 token thời gian**.
Bên video, CLS của mỗi frame `image_lowdim_norm.view(n,16,203,128)[:,:,0,:]` cho
**16 token thời gian**.

→ Bài toán OT chỉ là **16 × 32 mỗi sample**. Rất rẻ, chạy 12 lần/forward vẫn không đáng kể.

## 2. Vì sao là *Unbalanced* OT (điểm cần verify)

Balanced OT ép mọi khối lượng phải được vận chuyển: tổng mass = 1, mọi audio slot
bắt buộc được match. Trong DFER in-the-wild điều đó là sai — có im lặng, nhạc nền,
tiếng người ngoài khung hình, và các frame không có bằng chứng audio nào.

Mình đã kiểm chứng bằng số trên chính solver sẽ dùng (16 frame × 32 audio slot,
cố ý làm 10/32 audio slot thành nhiễu ở xa mọi frame):

| | mass rơi vào 10 slot nhiễu | mass vào slot tốt |
|---|---|---|
| Balanced (τ→∞) | **0.3125** (bị ép) | 0.6875 |
| UOT (τ=0.1) | **0.0000** | 0.1271 |

Và ở giới hạn τ→∞ solver tái tạo đúng Sinkhorn cân bằng (tổng mass = 1.0000,
marginal đều = 1/16). Đó là claim để verify, và cũng là ablation quan trọng nhất.

## 3. Code vào file nào — 5 file

> **Trạng thái: đã tích hợp xong.** Toàn bộ patch dưới đây đã được áp dụng vào repo
> và verify bằng torch 2.2.0. Phần này giữ lại để bạn biết chính xác cái gì đã đổi
> ở đâu, và để tự sửa/rollback (`git diff`, `git checkout -- <file>`).

| File | Trạng thái | Nội dung |
|---|---|---|
| `models/uot.py` | ✅ đã tạo | Solver UOT log-domain + module `UOTFusion` |
| `models/Generate_Model.py` | ✅ đã sửa | import, khởi tạo `ModuleList`, gọi trong vòng lặp (+20 dòng) |
| `main.py` | ✅ đã sửa | argparse flag, `requires_grad`, list `shutil.copyfile` (+17 dòng) |
| `evaluate.py` | ✅ đã sửa | cùng bộ flag (nếu thiếu sẽ dựng baseline và load checkpoint hỏng) (+10 dòng) |
| `train_MAFW_uot.sh`, `train_DFEW_uot.sh` | ✅ đã tạo | script chạy |

### 3.1 `models/uot.py` (đã có sẵn trong repo)

Hai thành phần:

- `unbalanced_sinkhorn_log(C, eps, tau, n_iters)` — scaling iteration ở log-domain
  cho UOT với KL relaxation hai chiều:
  `min_π <C,π> + ε·KL(π|a⊗b) + τ·KL(π1|a) + τ·KL(πᵀ1|b)`.
  Unrolled, differentiable, ổn định số. τ→∞ ⇒ Sinkhorn cân bằng.
- `UOTFusion` — 1 instance / block:
  1. lấy 16 token video (CLS mỗi frame) + 32 token audio (pool theo tần số),
  2. cost `C = 1 − cosine`, `[n,16,32]`,
  3. giải UOT ⇒ plan `π`,
  4. `a2v = row_norm(π) @ audio` rồi **nhân lại với row-mass** (frame nào không
     match được audio thì nhận update nhỏ — đây chính là phần "unbalanced" có tác dụng),
  5. `v2a` đối xứng,
  6. qua `Linear` + **gate khởi tạo 0** (`tanh(0)=0`).

Gate zero-init là chủ ý: **ở step 0 model ra kết quả trùng khít baseline**, nên
training không bị phá và mọi cải thiện đo được đều quy về UOT.

### 3.2 Patch `models/Generate_Model.py`

Thêm import ở đầu file:

```python
from models.uot import UOTFusion
```

Thêm vào **cuối** `__init__` (sau dòng `assert len(self.audio_model.blocks) == ...`):

```python
        self.use_uot = getattr(args, 'use_uot', False)
        if self.use_uot:
            self.uot_fusion = nn.ModuleList([
                UOTFusion(dim=128,
                          n_frames=16,
                          n_image=self.n_image,
                          n_audio_t=32,
                          n_audio_f=8,
                          eps=getattr(args, 'uot_eps', 0.05),
                          tau=getattr(args, 'uot_tau', 1.0),
                          n_iters=getattr(args, 'uot_iters', 10),
                          detach_plan=getattr(args, 'uot_detach', False))
                for _ in range(len(self.image_encoder.blocks))])
```

Trong `forward`, **giữ nguyên 2 dòng mean-pool cũ** và chèn ngay sau chúng:

```python
            if self.use_uot:
                a2v, v2a = self.uot_fusion[ii](image_lowdim_norm, audio_lowdim, n, t)
                image_lowdim_norm2 = image_lowdim_norm2 + a2v   # [n*t,1,128] broadcast
                audio_lowdim2      = audio_lowdim2 + v2a        # [n,1,128]   broadcast
```

Giữ nhánh mean-pool cũ là có chủ đích: UOT trở thành phần **cộng thêm** có gate,
không phá đường đi đã được pretrain. Nếu sau này muốn ablation "UOT thay thế hẳn
mean-pool" thì đổi `image_lowdim_norm2 = image_lowdim_norm + a2v`.

### 3.3 Patch `main.py`

Trong `parse_args()`:

```python
    parser.add_argument('--use-uot', action='store_true')
    parser.add_argument('--uot-eps', type=float, default=0.05)
    parser.add_argument('--uot-tau', type=float, default=1.0)
    parser.add_argument('--uot-iters', type=int, default=10)
    parser.add_argument('--uot-detach', action='store_true')
```

Trong khối `requires_grad` (thêm vào cuối chuỗi `if`):

```python
        if "uot_fusion" in name:
            param.requires_grad = True
```

> Lưu ý cách freeze hiện tại: vòng đầu bật `True` cho tất cả, rồi tắt mọi tên chứa
> `image_encoder` / `audio_model`, rồi bật lại theo keyword. Vì `uot_fusion` nằm ở
> cấp `GenerateModel` (không chứa `image_encoder`), nó **không bị tắt** — dòng trên
> chỉ để tường minh. Chỉ cần đừng đặt module UOT bên trong `image_encoder`.

Thêm `'models/uot.py'` vào cả hai list `shutil.copyfile` (DFEW và MAFW) — thư mục
đích `code/models/` đã được `os.makedirs` sẵn nên không cần sửa gì thêm. Việc này
giữ nguyên tính năng snapshot code vào log dir của repo.

### 3.4 Patch `evaluate.py`

Copy nguyên 5 dòng argparse ở 3.3 vào `parse_args()` của `evaluate.py`. Bắt buộc:
`evaluate.py` cũng gọi `GenerateModel(args=args)`, nếu thiếu flag nó dựng kiến trúc
baseline và `load_state_dict` sẽ vỡ khi nạp checkpoint có UOT.

### 3.5 Script chạy

```bash
cp train_MAFW.sh train_MAFW_uot.sh   # rồi thêm các flag UOT + đổi --exper-name
cp train_DFEW.sh train_DFEW_uot.sh
```

Nội dung thêm vào cuối mỗi script:

```bash
--use-uot \
--uot-eps 0.05 \
--uot-tau 1.0 \
--uot-iters 10 \
--exper-name UOT_tau1.0_eps0.05 \
```

## 4. Có phải train lại không? — **Có, bắt buộc.**

Ba lý do độc lập:

1. **Có tham số mới.** Mỗi block thêm `proj_a2v`, `proj_v2a` (2×128×128), 2 LayerNorm,
   2 gate ⇒ ~34k params/block × 12 ≈ **0.4M tham số mới**. Checkpoint công bố của
   tác giả không có chúng.
2. **Ngay cả bản UOT không tham số cũng phải train lại.** Fusion nằm ở *mọi block*,
   nên activation của toàn bộ 12 tầng đổi. Adapter/prompt đã pretrain được tối ưu
   cho phân phối activation cũ ⇒ nạp thẳng checkpoint cũ cho kết quả rác.
3. **Bản chất nhiệm vụ là kiểm chứng.** Muốn nói "UOT giúp cải thiện" thì cần
   baseline và UOT chạy **cùng seed (đang hardcode `seed=1`), cùng fold, cùng số epoch**.
   Không thể so với số trong paper vì dataset các bạn preprocess lại rồi.

### Ngân sách compute

Ma trận đầy đủ = 2 dataset × 2 config (baseline / UOT) × 5 fold × 25 epoch = **20 run**.
Đó là rất nhiều. Đề nghị làm theo thứ tự:

1. **Fold 1, MAFW, baseline** — 25 epoch. Lấy con số nền trên data của các bạn.
2. **Fold 1, MAFW, UOT (τ=1.0)** — 25 epoch. So sánh UAR/WAR.
3. Nếu có cải thiện → chạy nốt fold 2–5 cho cả hai config, rồi lặp cho DFEW.
4. Ablation τ ∈ {∞ (balanced), 1.0, 0.1} và ε ∈ {0.01, 0.05, 0.1} — **chỉ trên fold 1**.

Đo thời gian thật trước khi cam kết: chạy fold 1 và xem dòng `An epoch time` trong
`log/.../log.txt` sau epoch đầu, rồi nhân 25.

> **Cách tiết kiệm ~50% compute nếu cần:** vì gate zero-init làm model lúc đầu
> trùng baseline, có thể warm-start UOT run từ checkpoint baseline
> (`load_state_dict(..., strict=False)`) rồi chỉ train thêm ~8–10 epoch. `main.py`
> hiện **chưa có** cờ `--resume`, cần thêm ~10 dòng. Nói mình nếu muốn làm.

## 5. Thứ tự chạy

```bash
# 0. chuẩn bị data + checkpoint  -> xem SETUP.md
python tools/check_data.py --annotation annotation/MAFW_set_1_train_faces.txt --n 300

# 1. smoke test (bắt buộc, ~1 phút) - phải in "SMOKE TEST PASSED"
python tools/smoke_test.py --dataset MAFW --use-uot

# 2. baseline fold 1
./train_MAFW.sh

# 3. UOT fold 1
./train_MAFW_uot.sh
```

`tools/smoke_test.py` kiểm tra luôn 2 điều quan trọng nhất:
gradient có chảy tới module UOT không, và `max|baseline − uot| < 1e-4` tại thời điểm
khởi tạo (xác nhận gate zero-init hoạt động).

## 6. Hai cái bẫy trong `dataloader/video_dataloader.py` — đọc trước khi bỏ data lên server

Đây là rủi ro lớn nhất khi chuyển data từ Kaggle sang server, **không liên quan UOT
nhưng sẽ làm job chết**:

```python
if "clip_224x224" in video_frames_path[p]:            # nhánh DFEW
    ... .replace('clip_224x224','raw_wav') + '/' + str(int(...)) + '.wav'
elif "mfaw" in video_frames_path[p]:                  # nhánh MAFW - CHÍNH TẢ "mfaw"
    ... .replace('clips_faces','clips_wav') + '/' + ... + '.wav'
# KHÔNG có else -> `fbank` chưa được gán -> UnboundLocalError
```

1. **Đường dẫn MAFW bắt buộc chứa chuỗi `mfaw`** (viết sai chính tả, không phải `mafw`),
   và thư mục frame phải tên `clips_faces`, audio phải là `clips_wav`.
   Nếu data ở `/data/mafw/...` thì loader rơi vào nhánh nào cũng không khớp và crash.
2. **DFEW: `str(int(...))` cắt số 0 đầu.** Frame ở `clip_224x224/02522/` sẽ tìm audio ở
   `raw_wav/2522.wav` — **không phải** `02522.wav`.

`tools/check_data.py` mô phỏng đúng logic này và báo `UNMATCHED path` nếu dính bẫy.
Cách xử lý ở SETUP.md §3.
