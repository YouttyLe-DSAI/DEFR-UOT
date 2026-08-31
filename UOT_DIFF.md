# UOT đụng vào những phần nào của pipeline gốc

Trích thẳng từ `git diff 1c69b22 HEAD` (1c69b22 = bản MMA-DFER gốc).

---

## Tổng quan: 4 file, và 7 file gốc **không hề bị đụng**

```
models/uot.py              +146   ← FILE MỚI, toàn bộ phần UOT
models/Generate_Model.py    +26   ← nơi UOT được nối vào
main.py                     +59   ← trong đó chỉ ~10 dòng là UOT
evaluate.py                 +57   ← trong đó chỉ ~9 dòng là UOT
```

**Nguyên vẹn, không sửa một ký tự:**

```
dataloader/video_dataloader.py     dataloader/video_transform.py
models/models_vit.py               models/Temporal_Model.py
AudioMAE/audio_models_vit.py       train_MAFW.sh    train_DFEW.sh
```

Nghĩa là: cách đọc dữ liệu, hai encoder, temporal transformer, và script train baseline
đều **y hệt bản gốc**. Chạy `./train_MAFW.sh` là đang chạy đúng pipeline gốc.

## Trái tim: 3 điểm chạm trong `Generate_Model.py`

Đây là **toàn bộ** phần UOT trong pipeline. Không có gì khác.

**① Import** (dòng 8)
```python
from models.uot import UOTFusion
```

**② Khởi tạo** — cuối `__init__`, một `UOTFusion` cho mỗi block
```python
self.use_uot = getattr(args, 'use_uot', False)
if self.use_uot:
    self.uot_fusion = nn.ModuleList([
        UOTFusion(dim=128, n_frames=16, n_image=self.n_image,
                  n_audio_t=32, n_audio_f=8,
                  eps=..., tau=..., n_iters=..., detach_plan=...)
        for _ in range(len(self.image_encoder.blocks))])
```

**③ Forward** — chèn vào vòng lặp 12 block, ngay sau fusion mean-pool gốc
```python
if self.use_uot:
    a2v, v2a = self.uot_fusion[ii](image_lowdim_norm, audio_lowdim, n, t)
    image_lowdim_norm2 = image_lowdim_norm2 + a2v
    audio_lowdim2      = audio_lowdim2 + v2a
```

Điểm mấu chốt: **hai dòng fusion gốc được giữ nguyên**, UOT chỉ *cộng thêm* qua một gate
khởi tạo bằng 0. Nên khi không truyền `--use-uot`, hoặc ngay tại step 0 khi có `--use-uot`,
model cho ra kết quả **trùng khít** bản gốc.

## `main.py` / `evaluate.py`: chỉ 19/100 dòng thêm là UOT

81 dòng còn lại **không liên quan UOT**, và cần cho cả hai nhánh:

| Thêm | Vì sao |
|---|---|
| `weights_only=False` ×3 | torch ≥2.6 đổi mặc định; checkpoint nhúng `argparse.Namespace` → không có thì crash |
| `--folds` | chạy 1 fold thay vì cả 5 (giới hạn thời gian phiên) |
| `--resume` | warm-start từ checkpoint đã train |
| `--zero-audio` | đo nhánh audio có đóng góp thật không |

19 dòng UOT chỉ là: 5 cờ argparse ×2 file, cộng `if "uot_fusion" in name: param.requires_grad = True`,
cộng `'models/uot.py'` vào danh sách snapshot code.

## Hệ quả cho việc so sánh A/B

Vì UOT nằm gọn sau một cờ:

```bash
./train_MAFW.sh          # pipeline GỐC, không một thay đổi nào
./train_MAFW_uot.sh      # y hệt, chỉ thêm --use-uot --uot-*
```

`diff` hai script cho thấy chúng chỉ khác đúng các cờ UOT và `--exper-name`. Không có
biến ẩn nào khác.

## Kiểm chứng bằng số

`tools/smoke_test.py --use-uot` in ra:

```
max|baseline - uot| = 0.000e+00  OK
```

Tức là dựng model có UOT rồi tắt cờ đi thì output **giống hệt từng bit**. Đó là bằng
chứng UOT không phá gì của pipeline gốc.

Ngoài ra, `evaluate.py` nạp checkpoint tác giả với `strict=True` **thành công** khi
không bật `--use-uot` — xác nhận `state_dict` không đổi so với bản gốc.
