# Trạng thái công việc

Cập nhật: 2026-08-31 · Branch `feat/uot-fusion` · Mọi thứ đã push

Đọc file này trước khi tiếp tục ở một phiên/máy khác.

---

## Hai hướng UOT cùng tồn tại trong repo

Đây là điểm dễ nhầm nhất. Hai hướng **khác nhau hoàn toàn**, không phải hai phiên bản
của cùng một thứ.

| | A. UOT trong model | B. UOT post-hoc cross-corpus |
|---|---|---|
| Code | `models/uot.py`, `train_*_uot.sh` | `uot_crosscorpus/` |
| Ghép cặp gì | 16 frame ↔ 32 lát audio, trong 1 clip | mẫu DFEW ↔ mẫu MAFW, giữa 2 corpus |
| Cost matrix | audio vs visual, **khác metric space** | cùng encoder, **cùng space** |
| Cần train | có, 15–19 giờ GPU | **không** |
| Tài liệu | `UOT_INTEGRATION.md`, `UOT_ARCHITECTURE.md` | `uot_crosscorpus/README.md` |

**Hướng B là hướng đang theo đuổi.** Hướng A giữ lại theo yêu cầu, chưa xoá.

Lý do chuyển: nhóm đo được OT giữa audio và visual chỉ khôi phục **6.5%** tương ứng so
với **6.0%** đoán mò — hai modality không có metric chung nên cost matrix của hướng A là
nhiễu. Hướng B không dính vấn đề đó vì cả hai corpus đi qua cùng một checkpoint.

## Đã xác lập chắc chắn

- **Dữ liệu đúng.** Checkpoint tác giả chạy trên MAFW preprocess lại cho
  **UAR 43.86 / WAR 58.31** (5 fold), so với ~44.11 / 58.52 công bố. Lệch trong nhiễu.
- **Độ lệch giữa các fold rất lớn** — UAR 37.12 (fold 1) đến 48.81 (fold 3). Không bao
  giờ kết luận từ một fold. Luôn dùng `evaluate.py --folds 1 2 3 4 5`.
- `weights_only=False` là bắt buộc ở mọi `torch.load` — torch ≥2.6 đổi mặc định, mà mọi
  checkpoint ở đây nhúng `argparse.Namespace`.

## Thiết kế hiện tại của hướng B

Theo đặc tả hình do nhóm đưa. Không gian nhãn **7 lớp chung** (MAFW-7), bốn phương pháp:

| | |
|---|---|
| `none` | classifier đóng băng trên đặc trưng đích nguyên trạng |
| `prior_correction` | logit adjustment, prior đích ước lượng bằng SLD EM |
| `balanced_ot` | barycentric projection qua plan cân bằng → classifier |
| `unbalanced_ot` | như trên, plan nới biên KL |

**Tiêu chí thành công:** UOT phải thắng **cả ba**. Không thắng `prior_correction` thì
luận điểm "unbalanced" không có cơ sở.

### Hai quyết định dễ bị vô tình phá

1. **`--source-prior` phải để `uniform`.** Lệch prior lớp chính là hiện tượng đang
   nghiên cứu, phải tới được solver. `target-matched` tự tay sửa nó trước và xoá gần hết
   lý do dùng bản unbalanced. (Ở thiết kế open-set trước đó thì ngược lại — `target-matched`
   mới đúng. Đừng lẫn hai thiết kế.)
2. **Nguồn và đích phải cùng một checkpoint.** Đặc trưng từ hai encoder khác nhau không
   cùng metric space. `dfew_dfew` ↔ `dfew_mafw11`, `mafw_mafw11` ↔ `mafw_dfew`.

## Việc tiếp theo

1. Sinh 20 file `.npz` — `./tools/extract_all.sh <ckpt-root> dumps av` (~2 giờ GPU)
2. Chạy `python -m uot_crosscorpus.batch --dumps-root dumps --out uot_results.csv` (~30 phút CPU)
3. Đọc `MEETS` / `FAILS` và bảng per-class recall

Chi tiết từng bước trên server: `SERVER.md`.

## Chưa kiểm chứng được

Toàn bộ số của hướng B tới giờ là từ **dữ liệu tổng hợp** dựng để test code. Chúng chứng
minh code chạy đúng và phân biệt được bốn phương pháp — **không** chứng minh gì về dữ
liệu thật.

Một quan sát đáng theo dõi: trên synthetic, recall của `disgust` sụp ở cả hai phương
pháp OT (30–41% so với 96% của `none`), vì disgust chỉ chiếm 1.2% tập nguồn nên transport
không đủ khối lượng cấp cho lớp đó. Nếu lặp lại trên dữ liệu thật thì đó là kết quả đáng
báo cáo dù UOT thắng hay thua.

## Vướng mắc đã biết

- Bốn dataset MAFW/DFEW **không thuộc tài khoản `tunalmt`** — của thành viên khác, share
  cho. API cần đúng `<chủ-sở-hữu>/<slug>`, lấy từ URL trang dataset. Xem `SERVER.md` §2.3.
- `main` vẫn ở commit gốc `1c69b22`. Mọi thứ nằm ở `feat/uot-fusion`. Clone kiểu thường
  sẽ không thấy gì.
