# Bắt đầu từ đây — chạy ablation 4 nhánh trên server

Cập nhật **01/09/2026** · nhánh `feat/uot-fusion`

File này thay thế phần "Việc tiếp theo" của `HANDOFF.md`. Đọc hết trước khi chạy
bất cứ thứ gì — sắp tiêu **100–200 giờ GPU**.

---

## 0. Điều đã thay đổi so với `HANDOFF.md`

`HANDOFF.md` ghi *"Hướng B là hướng đang theo đuổi"* (UOT hậu kiểm liên tập).
**Không còn đúng.** Hướng B đã được đo đầy đủ ngày 01/09 và **âm tính**:

| thí nghiệm hậu kiểm | kết quả |
|---|---|
| OT/UOT chuyển đặc trưng + chiếu trọng tâm | disgust 10,97 → **0,31** |
| JCPOT ước lượng tiên nghiệm, 5 không gian biểu diễn | không cái nào có ý nghĩa |
| OSDA-OT phát hiện lớp lạ, 5 không gian | AUROC 0,51 so với 0,686 của `energy` |
| Partial OT | dưới mức đoán bừa |
| **BBSE** (không dùng OT) | **disgust +17,22, t = 7,64** |

Sáu thí nghiệm OT độc lập, không cái nào dương. Nguyên nhân chung: **không gian
đặc trưng 512 chiều không mang thông tin OT cần**. Thứ duy nhất chạy được dựa vào
**đầu ra bộ phân loại**, không phải hình học đặc trưng.

Số liệu và script nằm trên máy Mac: `KLTN_FA26/Results/label_shift.py`,
`openset_ot.py`, `ot_variants.py` và ba file `*_results.json`. Đã kiểm chứng chéo
ở cả hai độ phân giải 224 và 160 — mọi kết luận lặp lại.

**Nên giờ chạy hướng A** (UOT trong model, `models/uot.py`) dưới dạng ablation 4
nhánh. Hướng A **chưa bị bác bỏ**: chỉ dấu "6,5% so với 6,0%" trong `HANDOFF.md`
đo *chất lượng kế hoạch vận chuyển*, không đo *độ chính xác cuối* — hai câu hỏi
khác nhau. Ablation này trả lời câu thứ hai.

---

## 1. Bốn nhánh, và mỗi nhánh trả lời câu gì

```bash
                                                 # 1. baseline: không cờ UOT
--use-uot --uot-mode attn  --uot-tau 1.0         # 2. attention
--use-uot --uot-mode uot   --uot-tau 1e6         # 3. OT cân bằng
--use-uot --uot-mode uot   --uot-tau 1.0         # 4. UOT
```

| so sánh | trả lời |
|---|---|
| 2 vs 1 | thêm 402.456 tham số có ích không |
| 3 vs 2 | **cấu trúc vận chuyển** có ích, hay chỉ là thêm dung lượng |
| 4 vs 3 | **nới lỏng biên** có ích — tức chữ "unbalanced" có căn cứ |

**Bỏ nhánh 2 là mất bài.** Đó là nhánh phản biện sẽ hỏi đầu tiên. Nó dùng softmax
theo hàng trên *cùng ma trận chi phí*, cùng `norm`/`proj`/gate; softmax không có
tham số nên số tham số **khớp chính xác**.

---

## 2. Chạy

```bash
tmux new -s ab                      # bắt buộc: mất SSH không được mất run

# máy 3090
./run_ablation.sh 1 2

# máy 4090
./run_ablation.sh 3 4 5

# muốn theo dõi từ xa, gom cả hai máy một dashboard
WANDB=1 ./run_ablation.sh 1 2
```

Script tự chạy hai cổng kiểm tra, tự chặn nếu có `.py` chưa commit, tự bỏ qua run
đã xong (nên **chạy lại sau khi bị ngắt là an toàn**), và in git SHA vào mỗi run.

### Vì sao chia theo fold chứ không theo nhánh

3090 là `sm_86`, 4090 là `sm_89`. Thứ tự rút gọn dấu phẩy động khác nhau nên cùng
seed vẫn ra số hơi lệch. Chia theo nhánh (*"3090 chạy UOT, 4090 chạy baseline"*)
làm **nhánh trộn lẫn với máy** — không tách được nguyên nhân. Bốn nhánh của một
fold nằm cùng máy thì phép so là ghép cặp trong fold, chênh lệch do máy triệt tiêu.

**Một nhánh git, một SHA, hai máy checkout cùng SHA.** Đừng tạo hai nhánh cho hai
máy — đó là hai bản mã khác nhau.

---

## 3. Phải chạy trước, chưa ai chạy

Ba việc này **chưa làm được trên máy Mac** (thiếu backbone và `timm`):

```bash
# 1. Đồng nhất gate-0, chạy CẢ HAI chế độ
python3 tools/smoke_test.py --use-uot --uot-mode uot
python3 tools/smoke_test.py --use-uot --uot-mode attn
#    phải thấy:  max|baseline - uot| = 0.000e+00

# 2. Dữ liệu
python3 tools/check_data.py --annotation annotation/DFEW_set_1_train.txt --n 300
#    phải thấy:  missing .wav = 0   và   frame count off = 0

# 3. Đo một epoch để chốt lịch (ước lượng hiện tại chưa kiểm)
#    3090: ~23 phút/epoch ở 160 (đo được)   4090: ~13 phút (suy ra, CHƯA đo)
```

Trong lúc epoch đầu chạy, mở `watch -n2 nvidia-smi`:

- **GPU dưới 60%** → nghẽn dataloader, không phải GPU. Tăng `--workers`, và
  **chạy `tools/resize_frames.py` cho MAFW** nếu dùng MAFW. Commit `f6ae9af` đo
  được: giải mã JPEG là toàn bộ chi phí; ảnh nguồn 600px chậm gấp **12 lần** 224px.
- **VRAM sát ngưỡng** → giảm `--workers`. **Tuyệt đối không giảm `--batch-size`**,
  nó cố định ở 8 theo paper.

---

## 4. Rà soát 01/09 — 10 lỗi âm thầm đã vá, đừng phá lại

Một rà soát 13 agent tìm được 35 phát hiện, gộp còn 10 vấn đề. Hai cái **làm hỏng
phép so sánh ngay cả khi mọi thứ chạy trơn tru**. Tất cả đã vá.

| lỗi | đã vá bằng |
|---|---|
| Baseline lệch luồng dữ liệu so với 3 nhánh UOT | `set_seed(seed, fold, epoch)` + `generator=` + `worker_init_fn` |
| Nhánh `attn` mang tín hiệu khối lượng qua `col_mass` | bỏ cả hai phép nhân khi `mode == 'attn'` |
| `--resume-training` nạp mù | đối chiếu 16 trường `args`, chặn thiếu `{fold}`, kiểm số fold |
| Ghi checkpoint không nguyên tử | `.tmp` → `fsync` → `os.replace` |
| `evaluate.py` bỏ `uot_tau/eps/iters` | đọc cả 5 trường từ checkpoint |
| MAFW `tree224` thiếu `clips_wav` | `retarget_annotations.py` dừng hẳn |
| `resize_frames` để lại clip cụt | `rmtree` + thoát mã ≠ 0 |
| `compare_runs` trộn fold | ghép cặp theo fold |
| `smoke_test` thiếu `--uot-mode` | đã thêm |
| `model_best.pth` | **đã bỏ** — xem mục 5 |

### Lỗi nặng nhất, để hiểu vì sao đừng đụng vào phần seed

`seed = 1` chỉ đặt **một lần lúc import**. Bật `--use-uot` thì model dựng thêm 12
`UOTFusion` × 2 `nn.Linear` = **24 lần rút RNG toàn cục** mà baseline không rút.
`DataLoader(shuffle=True)` lấy seed từ chính RNG đó → **thứ tự batch, frame được
chọn, augmentation đều khác nhau** giữa baseline và ba nhánh UOT, suốt 25 epoch ×
5 fold. Log vẫn ghi `seed=1`. Ba nhánh UOT khớp nhau, nên **chỉ đúng nhánh làm mốc
bị lệch** — tức đúng cái duy nhất không được phép lệch.

`tools/verify_seed.py` vừa tái hiện lỗi cũ vừa chứng minh bản vá. Chạy nó nếu ai
đó đụng vào phần seed.

---

## 5. Ràng buộc không được vi phạm

**Siêu tham số đúng paper**: 25 epoch, lr 1e-4 cosine về 0, batch 8, wd 1e-2,
AdamW, seed 1. Đã đối chiếu mục 4.2 của paper: đúng hết.

**Báo cáo epoch cuối, không phải best.** Paper mục 4.2 nguyên văn: *"report the
result of final checkpoint, i.e., at 25th epoch"*.

`model_best.pth` **đã bị bỏ**, và lý do quan trọng hơn chuyện đúng giao thức: repo
này **không có tập validation riêng** — `val_loader` dựng từ **tập test**, nên
`is_best` được chọn bằng cách nhìn vào tập test. Con số từ file đó là giá trị lớn
nhất qua 25 lần đo trên test. **Đừng khôi phục nó.** Đường cong vẫn nằm trong
`log.txt` (`The best accuracy: ...`) để chẩn đoán.

Số vào bài lấy từ `log.txt`, dòng `UAR:` / `WAR:` do `computer_uar_war` ghi.

---

## 6. Ba cạm bẫy cũ, vẫn còn nguyên

1. **Audio giữ sample rate gốc.** Bản 16 kHz làm tụt 2,59 UAR, không báo lỗi.
2. **Số frame**: MAFW ghi `số ảnh − 1`, DFEW ghi đủ số.
3. **Đường dẫn MAFW phải chứa chuỗi `mfaw`** — sai thì fbank ra toàn 0, im lặng.

Thêm một cái mới tìm được: đường dẫn audio **suy ra từ đường dẫn ảnh** bằng thay
chuỗi `clips_faces` → `clips_wav`. Cây `tree224` mới resize chỉ có ảnh, thiếu wav
→ fbank toàn 0 ở **cả bốn nhánh**. Đã chặn ở `retarget_annotations.py`.

---

## 7. Theo dõi

```bash
WANDB=1 ./run_ablation.sh 1 2      # gom hai máy một dashboard
TB=1    ./run_ablation.sh 1 2      # TensorBoard cục bộ
```

Nhóm theo `fold`, phân theo nhánh, tag kèm tên GPU. Mọi lời gọi đều nuốt lỗi —
mất mạng không làm đổ run 8 tiếng. **`log.txt` vẫn là nguồn sự thật.**

**Thứ đáng nhìn nhất là giá trị gate** (`gate/abs_mean`). Gate khởi tạo 0; tới
epoch 5 vẫn ~0 nghĩa là model đang học cách **bỏ qua nhánh UOT** — run đó chỉ là
baseline chạy lại. Dừng sớm, đỡ 20 epoch. Nhìn UAR không phân biệt được điều đó
với "UOT không giúp gì".

---

## 8. Đọc kết quả

```bash
grep -H '^UAR: ' log/*AB_*-set*-log/log.txt
python3 tools/compare_runs.py          # ghép cặp theo fold, in rõ uot_mode/uot_tau
```

Kết luận **luôn theo fold**, không bao giờ từ một fold đơn lẻ: `HANDOFF.md` ghi độ
lệch giữa các fold tới **12 điểm UAR**.

Nếu 3 ≈ 4 thì tính "unbalanced" không đóng góp gì — **và phải nói thẳng điều đó**.

---

## 9. Tài liệu khác

| file | nội dung |
|---|---|
| `SERVER.md` | thao tác chi tiết trên server |
| `UOT_ARCHITECTURE.md` | UOT nằm ở đâu trong pipeline, 5 bước bên trong |
| `UOT_DIFF.md` | UOT đụng vào những phần nào của mã gốc |
| `SETUP.md`, `KAGGLE.md` | cài đặt |
| `HANDOFF.md` | bối cảnh cũ — **mục "Việc tiếp theo" đã lỗi thời**, xem mục 0 |
