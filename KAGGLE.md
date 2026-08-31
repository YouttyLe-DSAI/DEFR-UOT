# Chạy pipeline trên Kaggle

Mục đích của Kaggle ở đây là **kiểm chứng pipeline chạy được**, không phải lấy số cuối
cùng (lý do ở §5). Xong thì chuyển sang server theo `SETUP.md`.

---

## 1. Vấn đề với bộ dataset đang attach

Cấu trúc thật (sau khi mở rộng cây trong sidebar):

```
mafw-faces-native-part1/mafw_faces_native_shard0/mfaw/clips_faces/<id>/*.jpg
mafw-faces-native-part2/mafw_faces_native_shard1/mfaw/clips_faces/<id>/*.jpg
<audio-slug>/mfaw/clips_wav/<id>.wav
dfew-preprocessed.../dfew_frames_part1..4/clip_224x224/<id>/<id>_<n>.jpg
dfew-preprocessed.../dfew_audio_native/raw_wav/<int>.wav
```

**Tin tốt:** đường dẫn đã chứa sẵn `mfaw` / `clips_faces` / `clip_224x224` → dataloader
**khớp đúng branch**. Phần đặt tên bạn làm là chuẩn.

**Tin xấu:** loader suy ra đường dẫn `.wav` **từ đường dẫn frame bằng phép thay chuỗi**,
mà frames và audio nằm ở **mount khác nhau**:

| | loader đi tìm wav ở | wav thực sự nằm ở |
|---|---|---|
| MAFW | `.../mafw_faces_native_shard0/mfaw/clips_wav/03747.wav` | `/kaggle/input/<audio-slug>/mfaw/clips_wav/03747.wav` |
| DFEW | `.../dfew_frames_part1/raw_wav/2522.wav` | `.../dfew_audio_native/raw_wav/2522.wav` |

Không trùng. Và **hai dataset hỏng theo hai kiểu khác nhau**:

- **DFEW — crash.** Nhánh DFEW không có fallback, `torchaudio.load` trên file không tồn
  tại sẽ ném lỗi. Khó chịu nhưng ít nhất bạn biết ngay.
- **MAFW — hỏng im lặng, nguy hiểm hơn nhiều.** Nhánh MAFW có
  `if not os.path.exists(wav): fbank = torch.zeros(512,128)`. Training chạy hết 25 epoch,
  loss giảm, accuracy có số — **nhưng modality audio đã chết hoàn toàn**, và toàn bộ so
  sánh UOT trở nên vô nghĩa. Không có một dòng cảnh báo nào.

> Lưu ý khi tự kiểm tra: sau chuẩn hoá `(fbank + 4.2677) / 9.1380`, tensor toàn 0 trở
> thành **hằng số ~0.467**, không phải 0. Nên dấu hiệu nhận biết audio chết là
> **`std == 0`**, không phải "toàn số 0". Cách kiểm tra rẻ nhất là `evaluate.py --zero-audio`:
> nếu bật cờ đó mà kết quả không đổi thì audio vốn đã không đóng góp gì.

Vấn đề thứ ba: frames bị chia thành nhiều mount (MAFW 2 shard, DFEW 4 part) nhưng
annotation chỉ ghi được một gốc đường dẫn.

Cả ba vấn đề được giải quyết bằng cây symlink ở §3 Cell 4, **không sửa dòng nào của
code baseline**.

## 2. Checkpoint: cẩn thận nhầm loại file

Repo cần đúng **2 file encoder**, tên bị hardcode trong `models/Generate_Model.py`:

| Cần | Là gì |
|---|---|
| `mae_face_pretrain_vit_base.pth` | encoder **thị giác** MAE-Face ViT-B |
| `audiomae_pretrained.pth` | encoder **âm thanh** AudioMAE ViT-B |

Model `modelMMA` chứa nhiều hơn thế, và không phải file nào cũng dùng để train:

| File trong mount | Thực chất | Dùng làm gì |
|---|---|---|
| `pretrained.pth` | encoder AudioMAE | ✅ đổi tên → `audiomae_pretrained.pth` |
| `mae_face_visualize_vit_base.pth` | MAE-Face **bản có decoder** | ⚠ không phải bản `pretrain` |
| `checkpoint/MAFW_224/fold*.pth` | model MMA-DFER **đã train xong** | ❌ không dùng train, dùng để verify |
| `checkpoint/DFEW_224/fold*.pth` | như trên | ❌ / verify |

> ⚠ `GenerateModel` nạp encoder bằng `strict=False`. Đưa nhầm file **không báo lỗi** —
> nó im lặng nạp gần như không có gì rồi train từ đầu, và bạn chỉ phát hiện sau khi
> accuracy thấp một cách khó hiểu. Dùng `tools/check_ckpt.py` để nhận diện file trước:

```bash
python tools/check_ckpt.py <mount>/pretrained.pth <mount>/checkpoint/mae_face_visualize_vit_base.pth
```

Script phân biệt bằng nội dung: `patch_embed.proj.weight` có 1 kênh → audio encoder,
3 kênh → vision encoder; có `decoder_*` → bản `visualize`; có `our_classifier`/
`temporal_net` → model đã train.

Về bản `visualize`: nó gồm cả decoder, khác bản `pretrain` mà repo yêu cầu. Encoder vẫn
nạp được dưới `strict=False`, nhưng **phải đọc dòng `Image checkpoint loading:`** in ra
lúc dựng model — `missing_keys` dài nghĩa là không nạp được gì đáng kể, khi đó tải đúng
`mae_face_pretrain_vit_base.pth` từ
[MAE-Face releases](https://github.com/FuxiVirtualHuman/MAE-Face/releases).

### Dùng checkpoint đã train của tác giả để kiểm chứng dataset

Đây là công dụng thật của `checkpoint/{MAFW,DFEW}_224/fold*.pth`: chạy model đã train
xong của tác giả trên dữ liệu **bạn tự preprocess**. Số ra gần số công bố ⇒ toàn bộ
đường ống dữ liệu đúng. Số thấp bất thường ⇒ có chỗ hỏng, và biết trước khi đốt GPU thì
rẻ hơn nhiều. Với DFEW fold 1 @224, README dataset của bạn ghi mốc **UAR 63.63 / WAR 73.99**.

```bash
python evaluate.py --dataset DFEW --fold 1 --img-size 224 \
    --checkpoint <mount>/checkpoint/DFEW_224/fold1_224.pth
```


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

> Torch của Kaggle mới hơn bản `2.2.0` mà `requirements.txt` ghim. Từ torch 2.6,
> `torch.load` đổi mặc định thành `weights_only=True`, và mọi checkpoint ở đây đều
> nhúng một `argparse.Namespace` nên sẽ bị từ chối:
> `WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace`.
> Repo đã truyền `weights_only=False` ở cả 5 chỗ gọi `torch.load`
> (`models/Generate_Model.py` ×2, `main.py`, `evaluate.py`, `tools/check_ckpt.py`),
> nên không cần làm gì thêm — chỉ cần nhớ `git pull` để có bản vá.

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

### Cell 7 — trích đặc trưng cho phân tích UOT

Bước GPU duy nhất mà phần UOT cần. Sinh 20 file `.npz` (4 tổ hợp checkpoint × tập test,
5 fold, phương thức `av`), khoảng 2 giờ.

```python
!./tools/extract_all.sh {CKPT}/ /kaggle/working/dumps av
```

Điều kiện: **cả hai** corpus đã dựng xong trong phiên này (Cell 4–5 chạy cho cả MAFW
lẫn DFEW), vì mỗi checkpoint được chạy trên cả hai tập test.

Xong thì chuyển sang `kaggle/UOT_CROSSCORPUS.ipynb` — từ đó trở đi là CPU.

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

Việc dựng dữ liệu và trích đặc trưng thì vừa một phiên. Phân tích UOT chạy trên CPU
nên không đụng tới quota GPU.

- **Trên Kaggle:** 1 fold, 3–5 epoch, cho cả baseline và UOT. Mục tiêu là trả lời
  *"pipeline có chạy đúng không, UOT có học không, loss có giảm không"*.
- **Phân tích UOT:** `uot_crosscorpus/README.md`, chạy trên CPU từ các file `.npz`.

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
