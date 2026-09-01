# Setup trên server — từ trắng tới có kết quả

Dữ liệu và checkpoint lấy từ Kaggle. Toàn bộ ~4 giờ, trong đó ~2 giờ là GPU.

---

## 1. Môi trường (5 phút)

```bash
git clone -b feat/uot-fusion https://github.com/YouttyLe-DSAI/DEFR-UOT.git
cd DEFR-UOT

conda create -n mmadfer python=3.10 -y
conda activate mmadfer
pip install -r requirements.txt
pip install kaggle
```

Nếu driver server không hợp với `torch==2.2.0`: cài torch khớp CUDA của server **trước**,
rồi `pip install -r requirements.txt` cho phần còn lại.

```bash
python -c "import torch, timm; print(torch.__version__, timm.__version__, torch.cuda.device_count())"
```

`timm` phải là **0.9.16**. Bản 1.x đổi API của `VisionTransformer` mà `models/models_vit.py`
kế thừa.

## 2. Lấy dữ liệu và checkpoint (~1 giờ)

### 2.1. Credential

Máy hiện tại **đã đăng nhập sẵn** (tài khoản `tunalmt`). Copy nguyên thư mục sang server,
không cần tạo token mới:

```bash
scp -r ~/.kaggle/ <user>@<server>:~/
ssh <user>@<server> 'chmod 600 ~/.kaggle/*'
```

> `~/.kaggle/` chứa **hai** file: `access_token` (credential đang được CLI dùng) và
> `kaggle.json`. Lưu ý `kaggle.json` ở máy này **không phải JSON** — nó là một dòng
> `export KAGGLE_API_TOKEN=KGAT_...`, tức token định dạng mới. Copy cả thư mục thì
> chắc chắn đúng dù CLI dùng file nào.

Kiểm tra trên server:

```bash
kaggle datasets list -m | head -3
```

### 2.2. Checkpoint — tải trực tiếp được

Model thuộc tài khoản của bạn, đã xác nhận tồn tại (`746417  tunalmt/modelmma  modelMMA`):

```bash
export DATA=/data/dfer && mkdir -p $DATA && cd $DATA
kaggle models instances versions download tunalmt/modelmma/pyTorch/default/1 -p .
tar -xzf *.tar.gz 2>/dev/null || unzip -o '*.zip'
```

### 2.3. Dataset — cần ref chính xác

⚠ Bốn dataset MAFW/DFEW **không thuộc tài khoản `tunalmt`**: `kaggle datasets list -m`
không liệt kê chúng, và tìm công khai cũng không ra (chúng là private của một thành viên
khác trong nhóm, được share cho bạn).

API cần đúng `<chủ-sở-hữu>/<slug>`, mà cái đó **không đọc được từ `/kaggle/input`**. Lấy
từ URL trang dataset trên Kaggle:

```
kaggle.com/datasets/<chu-so-huu>/<slug>
                    └───────────┬───────────┘
                        chính là ref cần dùng
```

Rồi:

```bash
kaggle datasets download -d <chu-so-huu>/<slug> --unzip -p .
```

Nếu tải báo `403`, dataset chưa được share với tài khoản của bạn — nhờ chủ sở hữu thêm
bạn vào phần *Collaborators*.

### 2.4. Đường tắt: khỏi tải dữ liệu thô

Phần phân tích UOT **chỉ cần 20 file `.npz`**, không cần ảnh gốc. Nếu chỉ định chạy
`uot_crosscorpus`, chạy bước trích đặc trưng **trên Kaggle** (nơi dữ liệu đã mount sẵn)
rồi mang kết quả về:

| | Dung lượng |
|---|---|
| Ảnh + audio thô của hai corpus | vài chục GB |
| 20 file `.npz` | **~100 MB** |

Trên Kaggle, chạy `kaggle/UOT_CROSSCORPUS.ipynb` Cell 2b → Save Version. Rồi trên server:

```bash
kaggle kernels output <chu-so-huu>/<ten-kernel> -p dumps/
```

Chỉ cần tải dữ liệu thô khi bạn định **train** (bước 9) hoặc chạy `evaluate.py` (bước 6).

## 3. Dựng cấu trúc thư mục (2 phút)

Dataloader suy ra đường dẫn `.wav` **từ đường dẫn frame bằng phép thay chuỗi**, mà các
shard nằm rời nhau. Script dựng cây symlink đúng layout, không tốn dung lượng:

```bash
python tools/kaggle_setup.py --diagnose --input-root $DATA     # xem cái gì đang có

python tools/kaggle_setup.py --dataset MAFW --auto --input-root $DATA --out $DATA/tree
python tools/kaggle_setup.py --dataset DFEW --auto --input-root $DATA --out $DATA/tree
```

Bắt buộc thấy `LAYOUT OK`. Đọc kỹ `clips w/o wav`: **phải = 0 với DFEW** (nhánh này
không có fallback), với MAFW phải nhỏ — loader thay wav thiếu bằng `torch.zeros(512,128)`,
nhiều quá thì audio chết âm thầm.

## 4. Trỏ annotation (2 phút)

```bash
cp -r annotation annotation.bak

python tools/retarget_annotations.py --dataset MAFW \
    --new-root $DATA/tree/mfaw/clips_faces --recount --drop-missing
python tools/retarget_annotations.py --dataset DFEW \
    --new-root $DATA/tree/dfew/clip_224x224 --recount --drop-missing

python tools/check_data.py --annotation annotation/MAFW_set_1_train_faces.txt --n 300
python tools/check_data.py --annotation annotation/DFEW_set_1_test.txt --n 300
```

Điều kiện đi tiếp: `missing folder 0`, `EMPTY folder 0`, **`UNMATCHED path 0`**,
`labels seen` = `0..10` (MAFW) / `0..6` (DFEW).

## 5. Checkpoint encoder (1 phút)

```bash
CKPT=$DATA/checkpoint
python tools/check_ckpt.py --find $DATA

cp $CKPT/pretrained.pth                       ./audiomae_pretrained.pth
cp $CKPT/mae_face_visualize_vit_base.pth      ./mae_face_pretrain_vit_base.pth
ls -la *.pth
```

Tên bị hardcode trong `models/Generate_Model.py`, phải đúng chính xác.

## 6. Kiểm chứng dữ liệu (~40 phút GPU) — cần dữ liệu thô

Chạy model **đã train của tác giả** trên dữ liệu bạn vừa dựng. Mọi thứ khác giống hệt
tác giả, nên chênh lệch chỉ có thể đến từ dữ liệu.

```bash
python evaluate.py --dataset MAFW --folds 1 2 3 4 5 --img-size 224 --workers 8 \
    --checkpoint "$CKPT/MAFW_224/fold{fold}_224.pth"
```

So dòng `mean over 5 folds` với số công bố (MAFW 11 lớp ≈ 44.11 / 58.52). Độ lệch giữa
các fold tới ~12 điểm UAR nên **một fold đơn lẻ không so được**.

Nếu thấp bất thường, `--zero-audio` cho biết nhánh audio có đóng góp thật không.

## 7. Trích đặc trưng (~2 giờ GPU) — cần dữ liệu thô, hoặc dùng đường tắt §2.4

Bước GPU duy nhất mà phần UOT cần. Sinh 20 file `.npz`:

```bash
./tools/extract_all.sh $CKPT dumps av
```

Mỗi file chứa đặc trưng 512 chiều, logit, nhãn, **và trọng số classifier đóng băng** —
barycentric projection cần nó để phân loại. Script bỏ qua file đã có nên chạy lại được.

Cần **cả hai** corpus đã dựng ở bước 3–4, vì mỗi checkpoint chạy trên cả hai tập test.

## 8. Phân tích UOT (~30 phút CPU)

Không cần GPU từ đây.

```bash
python -m uot_crosscorpus.batch --dumps-root dumps --out uot_results.csv
```

In ra ma trận thí nghiệm: 4 phương pháp × 2 chiều × 5 fold, kèm per-class recall và
`paired t` trên các fold. Chi tiết tham số: `uot_crosscorpus/README.md`.

## 9. Train UOT từ đầu — A/B trong 7 lớp chung

Đây là thí nghiệm chính, và là lý do cần server: 25 epoch × 2 nhánh ≈ **33 giờ**, không
vừa giới hạn 12 giờ của Kaggle.

### 9.1. Sinh split MAFW-7

```bash
python tools/make_mafw7.py --split both
```

Bỏ 4 lớp chỉ MAFW có → **6.059 clip train / 1.517 clip test** mỗi fold.

### 9.2. Hai nhánh, khác nhau đúng một cờ

```bash
COMMON="--dataset MAFW --folds 1 --epochs 25 --batch-size 8 --workers 8 \
        --lr 1e-4 --weight-decay 1e-2 --print-freq 20 --temporal-layers 1 --img-size 224 \
        --num-classes 7 \
        --train-annotation ./annotation/MAFW_set_{fold}_train_faces7.txt \
        --test-annotation  ./annotation/MAFW_set_{fold}_test_faces7.txt"

# A — pipeline gốc
python main.py $COMMON --exper-name AB_BASE_mafw7_scratch 2>&1 | tee base.log

# B — thêm UOT
python main.py $COMMON --use-uot --uot-eps 0.05 --uot-tau 1.0 --uot-iters 10 \
    --exper-name AB_UOT_mafw7_scratch 2>&1 | tee uot.log
```

**Không có `--resume`** — adapter và classifier khởi tạo ngẫu nhiên, `lr 1e-4` theo paper.
Hai encoder vẫn đóng băng; đó là thiết kế của MMA-DFER, không liên quan warm-start.

Chiều cross-corpus: đổi `--test-annotation` sang `./annotation/DFEW_set_{fold}_test.txt`.
Train DFEW: `--dataset DFEW` và bỏ `--train-annotation` (DFEW vốn đã 7 lớp).

### 9.3. Đọc kết quả

```bash
python tools/compare_runs.py --log-root log \
    --base AB_BASE_mafw7_scratch --uot AB_UOT_mafw7_scratch
```

In UAR/WAR hai nhánh, hiệu số, và **bảng recall từng lớp** — cần cả bảng đó, vì các lớp
có thể lệch mạnh mà triệt tiêu nhau trong UAR.

### 9.4. Kiểm tra khi vừa khởi động

| Dòng | Phải thấy |
|---|---|
| `Loss` ở step 0 | **~1.95** (`= ln 7`) — đúng cho train từ đầu |
| nhánh B so với nhánh A | **cùng loss step 0** (gate UOT khởi tạo 0) |

Trước đó chạy smoke test một lần cho chắc:

```bash
python tools/smoke_test.py --dataset MAFW --num-classes 7 \
    --train-annotation ./annotation/MAFW_set_{fold}_train_faces7.txt --use-uot --batch-size 2
```

### 9.5. Checkpoint và chạy tiếp khi bị ngắt

Mỗi epoch ghi hai file vào `log/<run>/checkpoint/`:

| | |
|---|---|
| `model.pth` | epoch mới nhất — dùng để **chạy tiếp** |
| `model_best.pth` | epoch có val accuracy cao nhất |

Trước đây chỉ có `model.pth` và nó bị ghi đè mỗi epoch, nên bản tốt nhất mất ngay khi
một epoch sau tệ hơn. Giờ giữ cả hai.

Bị ngắt giữa chừng thì chạy tiếp bằng `--resume-training` (khác `--resume`: nó khôi phục
**đầy đủ** trọng số + optimizer + lịch learning rate + bộ đếm epoch):

```bash
python main.py $COMMON --exper-name AB_UOT_mafw7_scratch \
    --resume-training "log/<run-cu>/checkpoint/model.pth"
```

Khôi phục lịch LR là bắt buộc, không phải tuỳ chọn: `CosineAnnealingLR` giảm lr theo
đường cosine suốt 25 epoch. Nếu chỉ nạp trọng số rồi chạy lại, lr quay về đầu chu kỳ và
đó không còn là thí nghiệm ban đầu nữa. Đã kiểm chứng dãy lr của "chạy liền 25 epoch"
trùng khớp từng giá trị với "ngắt ở epoch 10 rồi resume".

**Báo cáo con số nào?** `computer_uar_war` ở cuối dùng `model.pth`, tức **epoch cuối**.
Muốn báo theo epoch tốt nhất thì đánh giá `model_best.pth` bằng `evaluate.py` — miễn là
**áp dụng cùng quy tắc cho cả hai nhánh**.

### 9.6. Ngân sách

| Tốc độ train | 1 epoch | 25 epoch / nhánh | Cả hai nhánh |
|---|---|---|---|
| 2 clip/s | 57 phút | 23,6 giờ | **47 giờ** |
| 3 clip/s | 40 phút | 16,6 giờ | 33 giờ |
| 5 clip/s | 26 phút | 11,0 giờ | 22 giờ |

Đo `An epoch time` trong `log.txt` sau epoch đầu rồi nhân lên — bảng trên chỉ là ước lượng.

Nhiều GPU thì chạy song song hai nhánh:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON --exper-name AB_BASE_mafw7_scratch &
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON --use-uot ... --exper-name AB_UOT_mafw7_scratch &
wait
```

Hai nhánh độc lập hoàn toàn nên song song không ảnh hưởng tính hợp lệ.

## Chạy nền

```bash
tmux new -s dfer
conda activate mmadfer
./tools/extract_all.sh $CKPT dumps av 2>&1 | tee extract.log
# Ctrl-b d để detach, tmux attach -t dfer để quay lại
```

## Checklist

- [ ] `torch.cuda.is_available()` True, `timm==0.9.16`
- [ ] `kaggle_setup.py` in `LAYOUT OK` cho **cả** MAFW lẫn DFEW
- [ ] `check_data.py`: `UNMATCHED path 0`, `EMPTY folder 0`
- [ ] 2 file `.pth` encoder đúng tên ở thư mục gốc
- [ ] `evaluate.py --folds 1 2 3 4 5` cho trung bình gần số công bố
- [ ] `dumps/` có đủ 20 file `.npz`
- [ ] `df -h .` còn ≥ 200 GB (chỉ cần nếu tải dữ liệu thô; đường tắt §2.4 thì ~100 MB)
