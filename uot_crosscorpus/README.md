# UOT cho thích nghi cross-corpus DFEW ↔ MAFW

Chạy trên các file `.npz` mà baseline đã dump sẵn. **CPU, không cần GPU, không cần POT.**

```
uot.py       solver OT log-domain (balanced = trường hợp riêng tau→∞)
data.py      nạp dump, không gian nhãn DFEW(7) ⊂ MAFW(11)
metrics.py   UAR / WAR / AUROC / FPR@TPR95
run.py       pipeline chính
```

## Lấy dumps ở đâu

Nếu chưa có sẵn từ kernel baseline, sinh lại bằng `tools/extract_features.py`.
Chỉ cần **20 file**, không phải cả 90:

```bash
./tools/extract_all.sh <checkpoint-root> dumps av      # ~2 giờ GPU
```

Đặc trưng lấy là đầu vào của `our_classifier` — tức đầu ra `temporal_net`, biểu diễn
hợp nhất 512 chiều ngay trước bộ phân loại. Bắt bằng forward hook, không sửa model.

Script **không** bọc `DataParallel`: nó nhân bản module nên hook đăng ký trên bản gốc
sẽ không kích hoạt, và trên phiên 2 GPU đặc trưng sẽ rỗng hoặc cũ. Checkpoint phát
hành được lưu qua DataParallel nên script tự bỏ tiền tố `module.`.

Cả hai corpus phải sẵn sàng cùng lúc, vì mỗi checkpoint chạy trên **cả hai** tập test.

## Chạy

```bash
python -m uot_crosscorpus.run \
    --source Results/baseline_B/dumps/dfew_dfew_fold1_av.npz \
    --target Results/baseline_A/dumps/dfew_mafw11_fold1_av.npz \
    --source-prior target-matched \
    --eps 0.05 --tau 1.0 --sweep
```

**Nguồn và đích bắt buộc cùng một checkpoint.** Đặc trưng từ hai encoder khác nhau
không cùng metric space, cost matrix thành vô nghĩa — đúng lý do bản audio↔visual
thất bại. Script cảnh báo nếu tiền tố tên file khác nhau.

| Chiều | source | target |
|---|---|---|
| DFEW → MAFW | `dfew_dfew_*` | `dfew_mafw11_*` |
| MAFW → DFEW | `mafw_mafw11_*` | `mafw_dfew_*` |

## ⚠ H1 như đề xuất viết đang bị nhiễu bởi lệch prior

Đề xuất dự đoán: *"Ở 0% lớp lạ, balanced và unbalanced phải trùng nhau."*

Điều đó **chỉ đúng khi prior của hai bên khớp**. Với biên đều (`--source-prior uniform`),
balanced OT khoá khối lượng mỗi lớp theo **tần suất nguồn**. Nếu lớp 5 chiếm 1.8% ở
nguồn nhưng 8.6% ở đích, balanced OT không thể gán đủ — trong khi UOT nới được cả
ràng buộc đó. Kết quả: gap khác 0 ngay tại 0% lớp lạ, và bị quy nhầm cho open-set.

Đo trên dữ liệu tổng hợp có cấu trúc giống thật:

| prior | gap tại **0%** lớp lạ | gap tại 100% |
|---|---|---|
| `uniform` | **+14.03 UAR** ← nhiễu, không phải open-set | +10.91 |
| `target-matched` | **+0.00 UAR** ✅ | +0.16 |

`target-matched` gán lại trọng số mỗi lớp nguồn theo tỉ lệ lớp đó trong **phần 7 lớp
chung của đích**. Tỉ lệ này không đổi khi sweep thêm mẫu lạ, nên chỉ còn đúng một biến
thay đổi — đúng thứ H1 tuyên bố đo.

Báo cáo baseline của nhóm đã ghi nhận lệch prior này (DFEW happy 20.9% / disgust 1.2%
so với MAFW), nên vấn đề sẽ xảy ra thật, không phải giả định.

**Dùng `--source-prior target-matched` cho hình chính.** Muốn giữ `uniform` thì phải
báo cáo cả hai và nói rõ gap gồm hai thành phần.

## ⚠ Chiều MAFW→DFEW cần `--source-label-space full`

Đề xuất mô tả chiều này là *partial adaptation*: "4 lớp nguồn thừa, không dùng tới".

Mặc định script **loại bỏ** 4 lớp đó khỏi nguồn — như vậy là **xoá bài toán thay vì
giải nó**, và balanced với unbalanced trở nên trùng nhau một cách tầm thường.

Thiết lập đúng là **giữ** chúng lại: balanced OT buộc phải tiêu khối lượng của
contempt/anxiety/helplessness/disappointment lên mẫu DFEW (sai, vì DFEW không có
các lớp đó), còn UOT được phép bỏ không dùng.

Đo trên dữ liệu tổng hợp, 5 fold:

| `--source-label-space` | gap unbalanced − balanced | paired t |
|---|---|---|
| `shared` (bỏ 4 lớp) | **+0.00** — tầm thường | — |
| `full` (giữ 4 lớp) | **+5.14 UAR**, dương cả 5/5 fold | **9.51** |

| Chiều | cấu hình đúng |
|---|---|
| DFEW → MAFW | mặc định (`shared`) — lớp lạ nằm ở **đích** |
| MAFW → DFEW | **`--source-label-space full`** — lớp thừa nằm ở **nguồn** |

## Tham số

| | |
|---|---|
| `--eps` | điều chuẩn entropy, tương đối so với thang của `C` |
| `--tau` | nới biên. Lớn → balanced. Nhỏ → vứt nhiều khối lượng hơn |
| `--metric` | `sqeuclidean` (mặc định) hoặc `cosine`; đều chuẩn hoá L2 trước |
| `--source-label-space` | `full` giữ lớp chỉ có ở nguồn (partial adaptation) |
| `--target-shared-only` | lọc đích còn 7 lớp. Đây là **điểm 0% của sweep**: không còn lớp lạ thì hai phương pháp phải trùng nhau, và điểm open-set không định nghĩa được |
| `--sweep` | quét tỉ lệ lớp lạ 0→100%, sinh số cho hình chính |

## Điểm open-set

`m_j = Σ_i γ_ij` — khối lượng mẫu đích nhận được. Mẫu thuộc lớp chưa từng thấy nhận
ít hơn. Script báo AUROC và FPR@TPR95 dùng `-m_j` làm điểm số. Balanced OT có tổng
khối lượng đúng bằng 1 nên `m_j` vẫn biến thiên, nhưng UOT mới là bản cho phép vứt
hẳn khối lượng.

## Đã kiểm chứng

- `tau → ∞` tái tạo Sinkhorn cân bằng (tổng mass = 1.0000, biên đều)
- AUROC: 1.0 khi tách hoàn hảo, 0.0 khi ngược, **0.5 khi điểm là hằng số**
- UAR/WAR đối chiếu tay trên ví dụ nhỏ
- Chạy trọn pipeline trên dump tổng hợp 512 chiều, đúng cấu trúc 7 ⊂ 11
- Phát hiện được H1 khi hiệu ứng tồn tại: gap 0.00 tại 0% → +0.16 khi có lớp lạ
