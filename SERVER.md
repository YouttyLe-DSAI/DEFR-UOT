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

## 2. Tải dữ liệu từ Kaggle (~1 giờ, tuỳ mạng)

Lấy API token: kaggle.com → Account → *Create New API Token* → được `kaggle.json`.

```bash
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

export DATA=/data/dfer            # đổi cho khớp server
mkdir -p $DATA && cd $DATA

# đổi <user>/<slug> theo đúng dataset của nhóm
kaggle datasets download -d <user>/mafw-faces-native-part1 --unzip -p .
kaggle datasets download -d <user>/mafw-faces-native-part2 --unzip -p .
kaggle datasets download -d <user>/mafw-native-rate-audio  --unzip -p .
kaggle datasets download -d <user>/dfew-preprocessed       --unzip -p .

# checkpoint (Kaggle Model, không phải Dataset)
kaggle models instances versions download tunalmt/modelmma/pyTorch/default/1 -p .
tar -xzf *.tar.gz 2>/dev/null || unzip -o '*.zip'
cd -
```

Không nhớ slug: `kaggle datasets list -m` liệt kê dataset của chính bạn.

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

## 6. Kiểm chứng dữ liệu (~40 phút GPU)

Chạy model **đã train của tác giả** trên dữ liệu bạn vừa dựng. Mọi thứ khác giống hệt
tác giả, nên chênh lệch chỉ có thể đến từ dữ liệu.

```bash
python evaluate.py --dataset MAFW --folds 1 2 3 4 5 --img-size 224 --workers 8 \
    --checkpoint "$CKPT/MAFW_224/fold{fold}_224.pth"
```

So dòng `mean over 5 folds` với số công bố (MAFW 11 lớp ≈ 44.11 / 58.52). Độ lệch giữa
các fold tới ~12 điểm UAR nên **một fold đơn lẻ không so được**.

Nếu thấp bất thường, `--zero-audio` cho biết nhánh audio có đóng góp thật không.

## 7. Trích đặc trưng (~2 giờ GPU)

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

## 9. (Tuỳ chọn) Train UOT trong model

Hướng khác, cần GPU thật sự — xem `UOT_INTEGRATION.md`.

```bash
tmux new -s uot
./train_MAFW_uot.sh          # hoặc bản warm-start với --resume
```

---

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
- [ ] `df -h .` còn ≥ 200 GB (frame của hai corpus)
