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
WORKERS=$(( $(nproc) - 2 ))
[ "$WORKERS" -lt 2 ] && WORKERS=2

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
python3 tools/verify_seed.py > /dev/null || { echo "HONG: verify_seed"; exit 1; }
echo "  verify_seed  DAT  (4 nhanh cung luong du lieu)"
python3 tools/verify_attn.py > /dev/null || { echo "HONG: verify_attn"; exit 1; }
echo "  verify_attn  DAT  (4 nhanh chi khac cach bien chi phi thanh trong so)"

# Ma nguon phai sach: hai may PHAI chay dung cung mot thu.
if [ -n "$(git status --porcelain -- '*.py' 2>/dev/null)" ]; then
    echo "DUNG LAI: co thay doi .py chua commit. Hai may phai cung mot SHA."
    git status --short -- '*.py'
    exit 1
fi
SHA=$(git rev-parse --short HEAD)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
echo "  SHA $SHA   GPU $GPU   workers $WORKERS"
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
