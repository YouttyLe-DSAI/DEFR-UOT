#!/bin/bash
# Chay ablation 4 nhanh cho cac fold duoc giao. Dung CHUNG cho moi may.
#
#   may 3090:  ./run_ablation.sh 1 2
#   may 4090:  ./run_ablation.sh 3 4 5
#
# MOT nhanh git, MOT commit, hai may checkout cung SHA. Viec chia nam o danh sach
# fold truyen vao, KHONG nam o ma nguon. Hai nhanh git cho hai may la hai ban ma
# khac nhau -- luc do khong ai chung minh duoc khac biet den tu UOT hay tu code.
#
# CHIA THEO FOLD, khong chia theo nhanh. 3090 la sm_86, 4090 la sm_89; thu tu rut
# gon dau phay dong tren hai kien truc khac nhau nen cung seed van ra so hoi lech.
# Neu chia theo nhanh ("3090 chay UOT, 4090 chay baseline") thi NHANH bi tron lan
# voi MAY va khong tach duoc nguyen nhan. Ca 4 nhanh cua mot fold nam cung mot may
# thi phep so la so ghep cap trong fold, chenh lech do may triet tieu het.
#
# THU TU FOLD-MAJOR: chay het 4 nhanh cua fold 1, roi moi sang fold 2. Sau ~4 run
# da co MOT fold hoan chinh de nhin, thay vi phai doi het moi co phep so dau tien.
# Bi ngat giua chung thi mat toi da mot fold do dang, khong phai mat het.

set -euo pipefail

FOLDS="${*:-}"
if [ -z "$FOLDS" ]; then
    echo "dung: $0 <fold> [fold ...]      vi du: $0 1 2"
    exit 1
fi

# --- Cau hinh chung. Doi o day thi doi cho CA BON nhanh, khong the lech. ---
DATASET=DFEW
IMG_SIZE=160
# So worker. Dem loi CPU cho ro rang: `$(( $(nproc) - 2 ))` neu nproc that bai se
# ra -2, roi dong ke tiep nang len 2 -- tuc tut tu 14 worker xuong 2 ma khong bao
# gi. Tren workload nghen dataloader do la mat gan het thong luong.
if [ -n "${WORKERS:-}" ]; then
    :                                        # nguoi dung tu dat, ton trong
elif NCPU=$(nproc 2>/dev/null) && [ -n "$NCPU" ]; then
    WORKERS=$(( NCPU > 4 ? NCPU - 2 : 2 ))
elif NCPU=$(getconf _NPROCESSORS_ONLN 2>/dev/null) && [ -n "$NCPU" ]; then
    WORKERS=$(( NCPU > 4 ? NCPU - 2 : 2 ))
else
    echo "DUNG LAI: khong dem duoc so CPU (thieu ca nproc lan getconf)."
    echo "  Dat tay roi chay lai:  WORKERS=12 $0 $FOLDS"
    exit 1
fi

# Theo doi. Deu tuy chon va chiu loi -- log.txt van la nguon su that.
#   WANDB=1 ./run_ablation.sh 1 2      gom ca hai may vao mot dashboard
#   TB=1    ./run_ablation.sh 1 2      TensorBoard cuc bo
TRACK=""
[ "${WANDB:-0}" = "1" ] && TRACK="$TRACK --wandb --wandb-project ${WANDB_PROJECT:-defr-uot}"
[ "${TB:-0}" = "1" ] && TRACK="$TRACK --tb"

# ============================================================================
# CONG KIEM TRA — khong dat thi khong chay. Re hon nhieu so voi phat hien sau
# 80 gio train.
# ============================================================================
echo "=== cong kiem tra ==="

# --- Cong cu. Thieu thi DUNG HAN, khong bao gio bo qua im lang. ---
# Ban truoc kiem git bang `$(git status ... 2>/dev/null)`: git khong co thi lenh
# that bai, stderr bi nuot, chuoi rong, `-n ""` sai, va cong CHO QUA. Mot cong
# an toan phai fail-closed; cai do fail-open, dung loai loi ma ca dot ra soat
# nay di tim.
need() {
    command -v "$1" > /dev/null 2>&1 || {
        echo "DUNG LAI: thieu '$1'"
        echo "  sua: $2"
        exit 1
    }
}
need git    "sudo apt update && sudo apt install -y git"
need python "PATH phai uu tien venv, vi du: export PATH=\$PWD/.venv310/bin:\$PATH"

# python phai la python CUA VENV, khong phai python he thong. Kiem bang thu that
# can dung chu khong doan qua duong dan -- ten venv moi may moi khac.
PY_PATH=$(command -v python)
python - <<'EOF' || {
import sys
import torch, timm            # noqa: F401  -- thieu la thoat khac 0
print('  python   {}  torch {}'.format(sys.version.split()[0], torch.__version__))
EOF
    echo "DUNG LAI: '$PY_PATH' khong nap duoc torch/timm."
    echo "  Day gan nhu chac chan la python he thong, khong phai venv."
    echo "  sua: export PATH=<duong-dan-venv>/bin:\$PATH   roi chay lai"
    exit 1
}
echo "  duong dan  $PY_PATH"

# Goi theo doi chi kiem khi thuc su bat -- thieu ma da bat co la loi cau hinh,
# khong phai chuyen de bo qua.
if [ "${TB:-0}" = "1" ]; then
    python -c 'import tensorboard' 2>/dev/null || {
        echo "DUNG LAI: TB=1 nhung chua co tensorboard."
        echo "  sua: pip install tensorboard     (hoac bo TB=1)"
        exit 1
    }
fi
if [ "${WANDB:-0}" = "1" ]; then
    python -c 'import wandb' 2>/dev/null || {
        echo "DUNG LAI: WANDB=1 nhung chua co wandb."
        echo "  sua: pip install wandb && wandb login     (hoac bo WANDB=1)"
        exit 1
    }
fi

python tools/verify_seed.py > /dev/null || { echo "HONG: verify_seed"; exit 1; }
echo "  verify_seed  DAT  (4 nhanh cung luong du lieu)"
python tools/verify_attn.py > /dev/null || { echo "HONG: verify_attn"; exit 1; }
echo "  verify_attn  DAT  (4 nhanh chi khac cach bien chi phi thanh trong so)"

# Ma nguon phai sach: hai may PHAI chay dung cung mot thu. Khong `2>/dev/null`
# o day -- loi cua git can duoc nhin thay.
DIRTY=$(git status --porcelain -- '*.py')
if [ -n "$DIRTY" ]; then
    echo "DUNG LAI: co thay doi .py chua commit. Hai may phai cung mot SHA."
    echo "$DIRTY"
    exit 1
fi
SHA=$(git rev-parse --short HEAD)
[ -n "$SHA" ] || { echo "DUNG LAI: khong doc duoc git SHA"; exit 1; }

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
echo "  SHA $SHA   GPU $GPU   workers $WORKERS"

# Ghi lai xuat xu ra dia. stdout cua tmux se troi mat; file thi con.
{
    echo "sha=$SHA"
    echo "gpu=$GPU"
    echo "python=$PY_PATH"
    echo "folds=$FOLDS"
    echo "img_size=$IMG_SIZE  dataset=$DATASET  workers=$WORKERS"
    echo "bat_dau=$(date '+%F %T')"
} >> RUN_MANIFEST.txt
echo

# ============================================================================
# BON NHANH. Viet DAY DU tung co, khong dung bien rut gon cho phan UOT.
# Ly do: chep mot dong roi quen mot co la tao ra ban sao cua nhanh khac, va no
# chay tron tru 25 epoch truoc khi ai do nhan ra. Dai dong o day la co chu y.
# ============================================================================
arm_flags() {
    case "$1" in
        baseline) echo "" ;;
        attn)     echo "--use-uot --uot-mode attn --uot-tau 1.0  --uot-eps 0.05 --uot-iters 10" ;;
        balanced) echo "--use-uot --uot-mode uot  --uot-tau 1e6  --uot-eps 0.05 --uot-iters 10" ;;
        uot)      echo "--use-uot --uot-mode uot  --uot-tau 1.0  --uot-eps 0.05 --uot-iters 10" ;;
        *) echo "nhanh la: $1" >&2; exit 1 ;;
    esac
}

da_xong() {
    # Da co run nao cua (nhanh, fold) nay chay den dong UAR chua?
    local ten="$1"
    for d in log/*"$ten"-set*-log; do
        [ -f "$d/log.txt" ] && grep -q '^UAR: ' "$d/log.txt" && return 0
    done
    return 1
}

TONG=0
for FOLD in $FOLDS; do
    for ARM in baseline attn balanced uot; do
        TEN="AB_${ARM}_f${FOLD}"
        if da_xong "$TEN"; then
            echo ">>> BO QUA $TEN (da co ket qua)"
            continue
        fi
        echo
        echo "============================================================"
        echo ">>> $TEN   fold $FOLD   $(date '+%F %T')"
        echo "============================================================"
        # shellcheck disable=SC2046
        python main.py \
            --dataset "$DATASET" \
            --folds "$FOLD" \
            --epochs 25 \
            --batch-size 8 \
            --lr 1e-4 \
            --weight-decay 1e-2 \
            --workers "$WORKERS" \
            --print-freq 50 \
            --temporal-layers 1 \
            --img-size "$IMG_SIZE" \
            $(arm_flags "$ARM") \
            $TRACK \
            --exper-name "$TEN"
        TONG=$((TONG + 1))
    done
done

echo
echo "=== xong $TONG run tren fold: $FOLDS  (SHA $SHA, $GPU) ==="
echo "Doc ket qua: grep -H '^UAR: ' log/*AB_*-set*-log/log.txt"
