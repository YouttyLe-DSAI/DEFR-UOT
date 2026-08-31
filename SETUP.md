# SETUP — đưa dataset lên server và chạy training

Viết cho tình huống: dataset MAFW + DFEW đã preprocess xong trên Kaggle, giờ
đưa về server để train. Làm tuần tự, **mỗi bước đều có lệnh kiểm tra**.

---

## 1. Môi trường

```bash
conda create -n mmadfer python=3.10 -y
conda activate mmadfer
pip install -r requirements.txt
```

`requirements.txt` ghim `torch==2.2.0 / torchaudio==2.2.0`. Nếu driver server không
hợp, cài torch khớp CUDA của server trước rồi mới `pip install -r requirements.txt`
cho phần còn lại. **Không cần cài thêm `POT`** — solver UOT trong `models/uot.py`
viết bằng PyTorch thuần.

Kiểm tra:

```bash
python -c "import torch, torchaudio, timm, einops; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 2. Checkpoint pretrain (2 file, đặt ở thư mục gốc repo)

| File | Tên bắt buộc | Nguồn |
|---|---|---|
| MAE-Face ViT-B | `mae_face_pretrain_vit_base.pth` | https://github.com/FuxiVirtualHuman/MAE-Face/releases |
| AudioMAE | `audiomae_pretrained.pth` | https://github.com/facebookresearch/AudioMAE |

Tên file được hardcode trong `models/Generate_Model.py` (`_build_image_model`,
`_build_audio_model`) — đặt sai tên là crash ngay lúc dựng model.

```bash
ls -la mae_face_pretrain_vit_base.pth audiomae_pretrained.pth
```

## 3. Bố trí dataset — phần dễ sai nhất

Loader chọn nhánh load audio bằng cách **tìm chuỗi con trong đường dẫn**
(`dataloader/video_dataloader.py`). Nên cấu trúc thư mục **không được đặt tùy ý**:

### DFEW

```
<DATA_ROOT>/dfew/clip_224x224/00001/00001.jpg ...   # frame, thư mục /clip
<DATA_ROOT>/dfew/raw_wav/1.wav                      # audio, TÊN KHÔNG CÓ SỐ 0 ĐẦU
```

`clip_224x224` phải xuất hiện nguyên văn trong path. Và vì loader tính tên wav bằng
`str(int("00001"))`, file audio của clip `00001` phải là **`1.wav`**, không phải `00001.wav`.

Nếu audio đang là `00001.wav`, đổi tên hàng loạt:

```bash
cd <DATA_ROOT>/dfew/raw_wav
for f in *.wav; do n="${f%.wav}"; mv -n "$f" "$((10#$n)).wav"; done
```

### MAFW

```
<DATA_ROOT>/mfaw/clips_faces/03747/00001.png ...    # LƯU Ý: "mfaw", không phải "mafw"
<DATA_ROOT>/mfaw/clips_wav/03747.wav                # giữ nguyên số 0 đầu
```

Chuỗi `mfaw` (sai chính tả trong code gốc) **bắt buộc** phải có trong path, thư mục
frame tên `clips_faces`, thư mục audio tên `clips_wav`.

Nếu thư mục đang tên `mafw`:

```bash
mv <DATA_ROOT>/mafw <DATA_ROOT>/mfaw
```

> Không muốn đổi tên thư mục? Sửa `dataloader/video_dataloader.py` dòng ~155 thành
> `elif "mfaw" in path or "mafw" in path:` và thêm `else: fbank = torch.zeros(512,128)`
> để không bao giờ crash vì `fbank` chưa gán. Cách đổi tên thư mục an toàn hơn vì
> không đụng vào code baseline.

### Giải nén từ Kaggle

```bash
mkdir -p <DATA_ROOT>
unzip -q mafw_preprocessed.zip -d <DATA_ROOT>/
unzip -q dfew_preprocessed.zip -d <DATA_ROOT>/
du -sh <DATA_ROOT>/*
find <DATA_ROOT>/mfaw/clips_faces -maxdepth 1 -type d | wc -l
find <DATA_ROOT>/mfaw/clips_wav -name '*.wav' | wc -l
```

## 4. Trỏ annotation về data của mình

10 file trong `annotation/` đang trỏ vào cluster của tác giả
(`/scratch/chumache/dfer_datasets/...`). Định dạng mỗi dòng:
`<thư_mục_frame> <số_frame> <label>`.

```bash
cp -r annotation annotation.bak     # luôn backup trước

python tools/retarget_annotations.py --dataset MAFW \
    --new-root <DATA_ROOT>/mfaw/clips_faces --recount --dry-run

# xem output ổn thì bỏ --dry-run
python tools/retarget_annotations.py --dataset MAFW \
    --new-root <DATA_ROOT>/mfaw/clips_faces --recount

python tools/retarget_annotations.py --dataset DFEW \
    --new-root <DATA_ROOT>/dfew/clip_224x224 --recount
```

`--recount` đếm lại số frame thật trên đĩa — cần thiết vì bước preprocess của các bạn
có thể ra số frame khác bản gốc, và loader dùng cột này để sample index
(sai ⇒ `IndexError` hoặc lấy nhầm frame). Thêm `--drop-missing` nếu chấp nhận bỏ
các clip thiếu.

## 5. Kiểm tra data trước khi train

```bash
python tools/check_data.py --annotation annotation/MAFW_set_1_train_faces.txt --n 300
python tools/check_data.py --annotation annotation/MAFW_set_1_test_faces.txt  --n 300
python tools/check_data.py --annotation annotation/DFEW_set_1_train.txt       --n 300
python tools/check_data.py --annotation annotation/DFEW_set_1_test.txt        --n 300
```

Yêu cầu để đi tiếp:

- `missing folder : 0`
- `frame count off: 0`
- `UNMATCHED path : 0` ← nếu khác 0 là dính bẫy §3, sửa xong mới chạy
- `labels seen` phải là `0..10` cho MAFW (11 lớp) và `0..6` cho DFEW (7 lớp)
- `missing .wav` nhỏ là chấp nhận được với MAFW (loader tự thay bằng `zeros(512,128)`),
  nhưng với DFEW thì **không có fallback** ⇒ phải bằng 0

## 6. Smoke test (bắt buộc, ~1 phút)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/smoke_test.py --dataset MAFW              # baseline
CUDA_VISIBLE_DEVICES=0 python tools/smoke_test.py --dataset MAFW --use-uot    # có UOT
CUDA_VISIBLE_DEVICES=0 python tools/smoke_test.py --dataset DFEW --use-uot
```

Phải thấy `SMOKE TEST PASSED`, và với `--use-uot` phải thấy:

- các tensor `uot_fusion.*` có `|grad|max` khác 0 → gradient chảy đúng
- `max|baseline - uot| = ...  OK` → gate zero-init đúng, model khởi đầu trùng baseline

Dòng `peak GPU memory` cho biết batch-size nào vừa VRAM. Batch 8 × 16 frame = 128 ảnh
qua ViT-B mỗi step, khá nặng. Nếu OOM: giảm `--batch-size` xuống 4 và giảm `--lr`
tương ứng (8→4 thì `--lr 7e-5`), hoặc giảm `--img-size 160`
(`n_image` tự tính `(img_size//16)**2` nên code vẫn chạy đúng).

## 7. Chạy training

**Luôn chạy trong `tmux`** — job kéo dài nhiều giờ, mất SSH là mất job.

```bash
tmux new -s mafw_base
conda activate mmadfer
CUDA_VISIBLE_DEVICES=0 ./train_MAFW.sh 2>&1 | tee run_mafw_baseline.log
# Ctrl-b d để detach, tmux attach -t mafw_base để quay lại
```

Lưu ý về script: `main.py` chạy **cả 5 fold liên tiếp trong một lần gọi**
(`for set in range(all_fold)`). Muốn chạy 1 fold trước để đo thời gian thì sửa tạm
`all_fold = 1` ở cuối `main.py`, hoặc đơn giản là để nó chạy và đọc kết quả fold 1
xong rồi quyết định.

Nhiều GPU: model bọc `torch.nn.DataParallel`, chỉ cần
`CUDA_VISIBLE_DEVICES=0,1` và tăng `--batch-size` theo số GPU.

## 8. Theo dõi

Mỗi lần chạy tạo `log/<DATASET>-<timestamp><exper-name>-set<k>-log/`:

```
log.txt          # loss/acc từng iteration, 'An epoch time', UAR/WAR cuối
log.png          # đường cong train/val
cn.png           # confusion matrix
checkpoint/model.pth
code/            # snapshot code của lần chạy đó
```

```bash
tail -f log/MAFW-*/log.txt
grep "An epoch time" log/MAFW-*set1*/log.txt | head -3     # ước lượng tổng thời gian
grep -E "UAR|WAR" log/MAFW-*/log.txt
watch -n 10 nvidia-smi
```

> Cảnh báo về đĩa: `save_checkpoint` ghi **mỗi epoch** (~1.5 GB/file, ghi đè cùng chỗ
> nên chỉ 1 file/fold). 5 fold × 2 config × 2 dataset ⇒ tính dư ~30–40 GB.
> `df -h .` trước khi chạy.

## 9. Đánh giá lại từ checkpoint

```bash
python evaluate.py --fold 1 --checkpoint log/MAFW-.../checkpoint/model.pth \
    --img-size 224 --dataset MAFW --use-uot
```

Cờ `--use-uot` **bắt buộc** phải khớp với lúc train, nếu không kiến trúc dựng ra khác
và `load_state_dict` sẽ lỗi.

## 10. Checklist trước khi bấm chạy

- [ ] `torch.cuda.is_available()` → True
- [ ] 2 file `.pth` pretrain đúng tên ở thư mục gốc
- [ ] `tools/check_data.py` sạch trên cả train và test của fold định chạy
- [ ] `annotation.bak/` đã backup
- [ ] `tools/smoke_test.py --use-uot` in `SMOKE TEST PASSED` + gate check OK
- [ ] `df -h .` còn ≥ 50 GB
- [ ] đang ở trong `tmux`
- [ ] `--exper-name` đặt tên phân biệt được baseline / UOT / giá trị τ
