# Chạy pipeline trên Kaggle

Mục đích của Kaggle ở đây là **kiểm chứng pipeline chạy được**, không phải lấy số cuối
cùng (lý do ở §5). Xong thì chuyển sang server theo `SETUP.md`.

---

## 1. Ba vấn đề với bộ dataset đang attach

Đối chiếu cấu trúc bạn đang có với logic của `dataloader/video_dataloader.py`:

```python
if   "clip_224x224" in frame_path:  wav = dir.replace('clip_224x224','raw_wav') + '/' + str(int(id)) + '.wav'
elif "mfaw"         in frame_path:  wav = dir.replace('clips_faces','clips_wav') + '/' + id + '.wav'
# KHÔNG có else  ->  biến `fbank` chưa gán  ->  UnboundLocalError
```

Mình đã chạy thử logic này trên đúng đường dẫn Kaggle của bạn:

| Đường dẫn thực tế | Kết quả |
|---|---|
| `/kaggle/input/mafw-faces-native-part1/mafw_faces_native_shard0/03747/00001.png` | ❌ **NO MATCH → crash** |
| `/kaggle/input/dfew-preprocessed-for-mma-dfer/dfew_frames_part1/02522/00001.jpg` | ❌ **NO MATCH → crash** |

**Vấn đề 1 — sai chuỗi nhận diện.** Thư mục của bạn là `mafw_faces_native_shard0`
(chứa `mafw`), nhưng code gốc viết sai chính tả và tìm `mfaw`. `mafw` ≠ `mfaw`.
Tương tự `dfew_frames_part1` không chứa `clip_224x224`.

**Vấn đề 2 — thiếu tên thư mục để `.replace()` hoạt động.** Kể cả khi khớp branch,
`.replace('clips_faces','clips_wav')` không làm gì vì đường dẫn không có `clips_faces`.

**Vấn đề 3 — frames và audio nằm ở các mount khác nhau (nghiêm trọng nhất).**
Loader suy ra đường dẫn `.wav` **từ đường dẫn frame bằng phép thay chuỗi**. Nhưng
frames ở `/kaggle/input/mafw-faces-native-part1/...` và `/…-part2/…`, còn audio ở
`/kaggle/input/mafw-native-rate-audio.../mfaw/clips_wav/`. Ba prefix khác nhau —
**không phép thay chuỗi nào bắc cầu được**. DFEW cũng vậy: 4 thư mục frames + 1 audio.

> ⚠️ **Chế độ hỏng âm thầm, nguy hiểm hơn crash.** Nhánh MAFW có fallback:
> nếu không tìm thấy `.wav` thì gán `fbank = torch.zeros(512,128)`. Nghĩa là nếu bạn
> chỉ sửa cho khớp chuỗi `mfaw` mà đường dẫn wav vẫn sai, **training sẽ chạy bình
> thường với audio toàn số 0** — modality audio chết hoàn toàn, và toàn bộ thí nghiệm
> UOT trở nên vô nghĩa mà không có lỗi nào báo. Script ở §3 kiểm tra đúng chỗ này.

## 2. Còn thiếu: checkpoint pretrain

Trong danh sách Input chưa thấy 2 file bắt buộc:

| File | Kích thước | Nguồn |
|---|---|---|
| `mae_face_pretrain_vit_base.pth` | ~1.3 GB | https://github.com/FuxiVirtualHuman/MAE-Face/releases |
| `audiomae_pretrained.pth` | ~1.2 GB | https://github.com/facebookresearch/AudioMAE |

Tải về máy → tạo một Kaggle Dataset (vd `mma-dfer-pretrained`) → Add Input.
Không có 2 file này thì `GenerateModel.__init__` chết ngay dòng `torch.load`.

## 3. Notebook — các cell

### Cell 1 — clone repo

```python
!git clone https://github.com/YouttyLe-DSAI/DEFR-UOT.git /kaggle/working/DEFR-UOT
%cd /kaggle/working/DEFR-UOT
!git log --oneline -3
```

Nếu repo để **private**, dùng token (lưu trong Kaggle *Secrets*, đừng hardcode):

```python
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GH_TOKEN")
!git clone https://{tok}@github.com/YouttyLe-DSAI/DEFR-UOT.git /kaggle/working/DEFR-UOT
```

> Nhớ `git push` từ máy local trước — các thay đổi UOT hiện đang nằm ở working tree
> chưa commit.

### Cell 2 — dependencies

```python
!pip install -q timm==0.9.16 einops==0.7.0 librosa==0.10.1
import torch, timm; print(torch.__version__, timm.__version__, torch.cuda.device_count())
```

**Bắt buộc ghim `timm==0.9.16`.** Image mặc định của Kaggle dùng timm 1.x, mà
`models/models_vit.py` kế thừa `timm.models.vision_transformer.VisionTransformer` —
API đã đổi giữa 2 major version. Torch/torchaudio thì dùng bản có sẵn của Kaggle,
đừng cài đè (sẽ mất bản CUDA).

### Cell 3 — xem chính xác cái gì đang được mount

```python
!python tools/kaggle_setup.py --diagnose
```

In cây thư mục 3 tầng của mọi dataset, tự đánh dấu chỗ nào là *clip folders* và chỗ
nào là *audio*. **Chép đúng đường dẫn từ output này vào Cell 4** — tên slug Kaggle
sinh ra không phải lúc nào cũng giống tên hiển thị.

### Cell 4 — dựng cây symlink

Đây là cách xử lý cả 3 vấn đề ở §1 **mà không đụng một dòng nào vào code baseline**:
ta dựng đúng cấu trúc mà loader mong đợi, bằng symlink (tốn ~0 dung lượng).

```python
# Khuyên dùng: tự dò theo nội dung, khỏi phải biết slug Kaggle
!python tools/kaggle_setup.py --dataset MAFW --auto --out /kaggle/temp/data
```

`--auto` phân loại theo **nội dung** chứ không theo tên: thư mục nào có ≥5 thư mục con
chứa ảnh → *frames root*; thư mục nào chứa `.wav` → *audio root*; rồi lọc theo
`mafw`/`mfaw` (hoặc `dfew`) để không lẫn hai dataset. Điều này quan trọng vì tên hiển
thị trong sidebar Kaggle là **tiêu đề** dataset, còn đường dẫn mount dùng **slug** —
hai thứ thường khác nhau.

Chỉ định tay khi cần (lấy đường dẫn từ Cell 3):

```python
!python tools/kaggle_setup.py --dataset MAFW \
  --frames /kaggle/input/mafw-faces-native-part1/mafw_faces_native_shard0 \
           /kaggle/input/mafw-faces-native-part2/mafw_faces_native_shard1 \
  --audio  /kaggle/input/<slug-that-cell-3-printed>/mfaw/clips_wav \
  --out    /kaggle/temp/data
```

Kết quả:

```
/kaggle/temp/data/mfaw/clips_faces/03747      -> shard0 hoặc shard1 (gộp 2 mount)
/kaggle/temp/data/mfaw/clips_wav/03747.wav    -> mount audio
```

Đường dẫn mới chứa `mfaw` ✓, chứa `clips_faces` ✓, và `.replace()` cho ra đúng
`clips_wav` ✓. Script tự chạy `verify` mô phỏng lại chính logic của dataloader và
in `LAYOUT OK` / `LAYOUT BROKEN`.

Tương tự cho DFEW (chú ý gộp cả 4 part, và script tự bỏ số 0 đầu khi đặt tên wav
vì loader gọi `str(int(id))`):

```python
!python tools/kaggle_setup.py --dataset DFEW --auto --out /kaggle/temp/data
```

Kiểm tra số `FRAMES` dò được khớp số shard đã attach (MAFW: 2, DFEW: 4).
Rồi đọc kỹ dòng `clips w/o wav`. Với DFEW con số này **phải bằng 0** (không có fallback).
Với MAFW phải nhỏ, nếu lớn thì audio đang chết âm thầm — xem cảnh báo ở §1.

> Dùng `/kaggle/temp` chứ không phải `/kaggle/working`: symlink không tốn dung lượng
> nhưng hàng chục nghìn file sẽ làm bước "Save Version" rất chậm. `/kaggle/temp` bị
> xoá mỗi session, dựng lại chỉ mất vài giây.

### Cell 5 — trỏ annotation vào cây vừa dựng

```python
!cp -r annotation annotation.bak
!python tools/retarget_annotations.py --dataset MAFW \
    --new-root /kaggle/temp/data/mfaw/clips_faces --recount --drop-missing
!python tools/check_data.py --annotation annotation/MAFW_set_1_train_faces.txt --n 300
!python tools/check_data.py --annotation annotation/MAFW_set_1_test_faces.txt  --n 300
```

Điều kiện đi tiếp: `missing folder: 0`, `frame count off: 0`, **`UNMATCHED path: 0`**,
`labels seen 0..10` (MAFW) hoặc `0..6` (DFEW).

`--recount` là bắt buộc: bộ preprocess của bạn gần như chắc chắn ra số frame khác
bản gốc, mà loader dùng cột đó để sample index.

### Cell 6 — copy checkpoint pretrain

```python
!cp /kaggle/input/mma-dfer-pretrained/mae_face_pretrain_vit_base.pth .
!cp /kaggle/input/mma-dfer-pretrained/audiomae_pretrained.pth .
!ls -la *.pth
```

Tên file bị hardcode trong `models/Generate_Model.py`, phải đúng chính xác.

### Cell 7 — smoke test (đừng bỏ qua)

```python
!python tools/smoke_test.py --dataset MAFW --use-uot --batch-size 2
```

Phải in `SMOKE TEST PASSED`, gradient của **gate** khác 0, và
`max|baseline - uot| ... OK`. Dòng `peak GPU memory` cho biết batch-size nào vừa VRAM.

### Cell 8 — train

```python
!python main.py --dataset MAFW --folds 1 --epochs 5 \
  --batch-size 4 --workers 2 --lr 7e-5 --weight-decay 1e-2 \
  --print-freq 20 --temporal-layers 1 --img-size 224 \
  --use-uot --uot-eps 0.05 --uot-tau 1.0 --uot-iters 10 \
  --exper-name KAGGLE_UOT
```

Baseline để so sánh: bỏ `--use-uot` và đổi `--exper-name KAGGLE_BASE`.

`--folds 1` là cờ mình mới thêm vào `main.py` — mặc định script gốc chạy **cả 5 fold
liên tiếp trong một process**, không thể xong trong giới hạn phiên của Kaggle.

Tham số đề xuất cho GPU Kaggle (T4 16GB): `--batch-size 4 --workers 2`
(Kaggle chỉ có 2–4 vCPU nên `--workers 8` sẽ nghẽn). Giảm batch 8→4 thì hạ lr
tương ứng 1e-4→7e-5.

## 4. Kết quả nằm ở đâu

```
/kaggle/working/DEFR-UOT/log/MAFW-<timestamp>KAGGLE_UOT-set1-log/
    log.txt      # loss/acc mỗi iteration, 'An epoch time', UAR/WAR
    log.png      # đường cong
    cn.png       # confusion matrix
    checkpoint/model.pth   (~1.5 GB, ghi đè mỗi epoch)
    code/        # snapshot code
```

Nằm trong `/kaggle/working` nên được giữ lại khi **Save Version**. Giới hạn 20 GB —
một checkpoint 1.5 GB thì thoải mái, nhưng đừng giữ nhiều run trong cùng notebook.

## 5. Giới hạn Kaggle — hãy đặt kỳ vọng đúng

| | |
|---|---|
| Thời gian tối đa 1 phiên GPU | ~12 giờ |
| Quota GPU | ~30 giờ/tuần |
| `/kaggle/working` | 20 GB, được lưu |
| `/kaggle/temp` | không lưu, xoá sau mỗi phiên |
| `/kaggle/input` | chỉ đọc |

Chạy đủ **5 fold × 25 epoch × 2 config** là **không khả thi** trên Kaggle. Vì vậy:

- **Trên Kaggle:** 1 fold, 3–5 epoch, cho cả baseline và UOT. Mục tiêu là trả lời
  *"pipeline có chạy đúng không, UOT có học không, loss có giảm không"*.
- **Trên server:** chạy thật theo `UOT_INTEGRATION.md` §4.

Với phiên chạy dài, dùng **Save Version → Save & Run All (Commit)** thay vì phiên
tương tác — phiên tương tác bị ngắt khi idle, commit thì chạy nền tới hết.

Ước lượng thời gian: xem `An epoch time` trong `log.txt` sau epoch đầu, nhân lên,
rồi mới quyết định đặt `--epochs` bao nhiêu cho vừa 12 giờ.

## 6. Checklist trước khi bấm Run All

- [ ] Đã `git push` các thay đổi UOT từ máy local
- [ ] Dataset checkpoint pretrain đã được tạo và Add Input (§2)
- [ ] `timm==0.9.16` đã cài (Cell 2)
- [ ] Cell 4 in `LAYOUT OK`, và `clips w/o wav` = 0 (DFEW) / nhỏ (MAFW)
- [ ] Cell 5 in `UNMATCHED path : 0`
- [ ] Cell 7 in `SMOKE TEST PASSED`
- [ ] `--folds 1` và `--epochs` đủ nhỏ để vừa 12 giờ
- [ ] `--exper-name` phân biệt được baseline / UOT
