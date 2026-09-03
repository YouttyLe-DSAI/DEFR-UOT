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
# DATASET va IMG_SIZE dat qua bien moi truong de khong phai sua file:
#   DATASET=MAFW ./run_ablation.sh 1 2
# So lop lay mac dinh cua main.py -- DFEW 7, MAFW 11 -- dung nhu paper bao cao.
DATASET="${DATASET:-DFEW}"
IMG_SIZE="${IMG_SIZE:-160}"
case "$DATASET" in
    DFEW|MAFW) ;;
    *) echo "DUNG LAI: DATASET phai la DFEW hoac MAFW, dang la '$DATASET'"; exit 1 ;;
esac
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

# MAFW co ba cai bay ma DFEW khong co. Ca ba deu hong IM LANG, va ca ba deu
# lam mat nhanh audio tren CA BON nhanh -- tuc pha ca loat ma so lieu van trong
# hop ly. Kiem trong 5 giay o day, thay vi phat hien sau 30 gio.
if [ "$DATASET" = "MAFW" ]; then
    ANN="annotation/MAFW_set_1_train_faces.txt"
    [ -f "$ANN" ] || { echo "DUNG LAI: khong thay $ANN"; exit 1; }
    DUONG_DAN=$(head -1 "$ANN" | cut -d' ' -f1)

    # 1. dataloader nhan dien MAFW bang chuoi 'mfaw' (viet sai chinh ta, co y).
    #    Thieu no thi di nhanh DFEW va fbank ra toan 0.
    case "$DUONG_DAN" in
        *mfaw*) ;;
        *) echo "DUNG LAI: duong dan MAFW khong chua chuoi 'mfaw':"
           echo "  $DUONG_DAN"
           echo "  dataloader nhan dien MAFW bang chuoi nay; thieu -> fbank toan 0."
           exit 1 ;;
    esac

    # 2. duong dan audio SUY RA tu duong dan anh. Thieu clips_wav -> fbank toan 0.
    WAV_DIR=$(dirname "$DUONG_DAN" | sed 's#clips_faces#clips_wav#')
    [ -d "$WAV_DIR" ] || {
        echo "DUNG LAI: khong thay thu muc audio $WAV_DIR"
        echo "  sua: ln -s <cay_goc>/mfaw/clips_wav $WAV_DIR"
        exit 1; }

    # 3. thu muc frame phai ton tai va so frame trong annotation phai khop.
    #    Quy uoc MAFW la 'so anh - 1'; lech la lay mau nham frame, im lang.
    read -r P N _ < "$ANN"
    [ -d "$P" ] || { echo "DUNG LAI: khong thay thu muc frame $P"; exit 1; }
    # -L la BAT BUOC: SERVER.md muc 4 dung data/stage/ toan bang symlink de gop
    # hai shard MAFW ma khong nhan doi 56 GB, nen $P la symlink. `find` KHONG di
    # vao symlink -- no coi symlink la file thuong va dem ra 0, roi cong bao
    # "annotation ghi 103 frame, tren dia co 0 anh" va chan MOI lan chay MAFW du
    # du lieu hoan toan dung.
    #
    # Kho thay o cho: `[ -d "$P" ]` ngay tren KHONG bat duoc, vi `-d` THI co di
    # theo symlink. No xac nhan thu muc ton tai, roi dong ke tiep dem ra 0.
    THUC=$(find -L "$P" -name '*.jpg' | wc -l | tr -d ' ')
    if [ "$N" -ne $(( THUC - 1 )) ]; then
        echo "DUNG LAI: annotation ghi $N frame, tren dia co $THUC anh."
        echo "  Quy uoc MAFW la 'so anh - 1', tuc phai ghi $(( THUC - 1 ))."
        echo "  Lech la lay mau nham frame, khong bao loi."
        exit 1
    fi
    echo "  MAFW    DAT  ('mfaw' co, clips_wav co, so frame = so anh - 1)"
fi

python tools/verify_seed.py > /dev/null || { echo "HONG: verify_seed"; exit 1; }
echo "  verify_seed  DAT  (4 nhanh cung luong du lieu)"
python tools/verify_attn.py > /dev/null || { echo "HONG: verify_attn"; exit 1; }
echo "  verify_attn  DAT  (4 nhanh chi khac cach bien chi phi thanh trong so)"

# --- Xuat xu: hai may PHAI chay dung cung mot ma nguon ---
#
# Do la dieu duy nhat cong nay can bao dam. Git SHA la cach goi gon nhat, nhung
# tren may lab khong co root thi khong cai duoc git, va lam ca thi nghiem dung
# lai vi mot cong cu ghi so la sai uu tien.
#
# Duong thoat: bam noi dung cac file nguon. No bao dam DUNG dieu can bao dam --
# tham chi chat hon SHA, vi bam noi dung THAT chu khong tin vao commit. Doi lai
# no khong biet gi ve lich su, nen chi dung khi that su khong co git.
#
# Khong tu dong roi ve bam: phai bat ALLOW_NO_GIT=1 tuong minh. Roi ve am tham
# la dung cai bay ma cong nay sinh ra de tranh.
# Bam CA TEN LAN NOI DUNG cua moi file .py, roi bam lai danh sach do. Bam rieng
# noi dung thoi thi doi ten hay di chuyen file se khong lam doi bam.
#
# PHAI ghim SHA-256 cho ca hai cong cu: `shasum` mac dinh la SHA-1 con
# `sha256sum` la SHA-256, nen hai may dung hai cong cu khac nhau se ra hai con
# so khac nhau tren CUNG mot ma nguon -- pha huy dung muc dich cua viec nay.
#
# Loai ./log/: main.py sao chep ma nguon vao log/<run>/code/ moi lan chay, nen
# khong loai thi bam doi sau run dau tien.
bam_nguon() {
    local h
    if command -v sha256sum > /dev/null 2>&1; then
        h="sha256sum"
    elif command -v shasum > /dev/null 2>&1; then
        h="shasum -a 256"
    else
        return 1
    fi

    # DANH SACH FILE phai la ma nguon cua repo, khong phai moi thu trong thu muc.
    # Ban truoc chi loai .git/, log/ va __pycache__, nen tren may dat venv TRONG
    # repo (dung nhu SERVER.md muc 2 huong dan) no bam 23.513 file .py, trong do
    # 44 la ma nguon con lai la .tools/ va .venv*/. Hai hau qua, ca hai deu pha
    # dung muc dich cua viec bam:
    #   - `pip install` mot goi bat ky la doi bam, ngay tren cung mot may
    #   - hai may khong bao gio khop, tru khi trung ca noi dung venv
    # Do chinh la chuyen da xay ra: 23.513 file ben 3090 so voi 46 ben 4090.
    #
    # LC_ALL=C la BAT BUOC. `sort` xep theo locale, ma repo co ca AudioMAE/ (hoa)
    # lan annotation/ (thuong): trong C thi hoa dung truoc, trong en_US.UTF-8 thi
    # khong. Cung mot tap file, hai thu tu, hai bam. Chuyen nay da gay bao dong
    # gia that -- 3090 (en_US.UTF-8) va 4090 (C.UTF-8) ra hai so khac nhau tren
    # cung mot commit sach tung byte, va mat mot vong dieu tra moi loai tru duoc
    # gia thuyet "hai may dang chay hai ban ma khac nhau". Cung khuon loi voi
    # shasum-vs-sha256sum da va o c8e8250, chi la tai xuat o tang sort.
    #
    # Ke ca .sh: arm_flags() nam trong file nay va giu co cua ca bon nhanh. Doi
    # mot dong o do la DOI THI NGHIEM, trong khi bam chi .py van khop hoan toan.
    # LIET KE, khong loai tru. Ban truoc dung `find` voi danh sach loai tru va
    # gap dung hai loi ma cach nay khong the co:
    #
    #   - danh sach loai tru la VO HAN. May 3090 co annotation.orig/ va
    #     env_local.sh -- ca hai do chinh SERVER.md huong dan tao -- nen find bat
    #     54 file trong khi git bat 51. Khong doan truoc duoc moi thu se xuat hien
    #     trong thu muc lam viec; nhung liet ke chinh xac thu gi thuoc thi nghiem
    #     thi duoc.
    #
    #   - TIEN TO ./ khac nhau giua hai nhanh. `git ls-files` cho 'main.py' con
    #     `find .` cho './main.py', ma sha256sum bam CA DUONG DAN. Cung mot cay ra
    #     hai bam khac nhau -- dung cai loi "hai may sinh hai dinh danh" ma ban va
    #     truoc dinh sua, tai tao lai o tang chuoi duong dan.
    #
    # Manifest duoc track trong repo nen may khong co git van biet file nao thuoc
    # ve no.
    [ -f SOURCE_MANIFEST ] || { echo "thieu SOURCE_MANIFEST" >&2; return 1; }
    grep -v '^#' SOURCE_MANIFEST | grep -v '^[[:space:]]*$' \
        | LC_ALL=C sort | tr '\n' '\0' | xargs -0 $h | $h | cut -c1-12
}

# Manifest phai theo kip ma nguon. Chi kiem duoc khi co git; may khong git tin
# vao ban da commit, dung nhu no tin vao chinh ma nguon.
kiem_manifest() {
    git rev-parse --git-dir > /dev/null 2>&1 || return 0
    local a b
    a=$(grep -v '^#' SOURCE_MANIFEST | grep -v '^[[:space:]]*$' | LC_ALL=C sort)
    b=$(git ls-files '*.py' '*.sh' | LC_ALL=C sort)
    [ "$a" = "$b" ] && return 0
    echo "DUNG LAI: SOURCE_MANIFEST khong khop voi ma nguon."
    diff <(echo "$a") <(echo "$b") | head -10
    echo "  sua: bash tools/make_manifest.sh && git add SOURCE_MANIFEST"
    return 1
}

# Bam noi dung tinh o CA HAI nhanh, khong chi nhanh khong-git.
#
# Ban truoc chia doi: may co git chi lay SHA, may khong git chi lay bam. Nen
# 4090 (co git) khong bao gio goi bam_nguon, 3090 (khong git) khong bao gio co
# SHA, va dong "HAI MAY PHAI CO CUNG CON SO NAY" chi in o phia khong-git -- phia
# kia khong biet la co gi de doi chieu. Phep so cheo may ma tai lieu dua vao la
# BAT KHA THI VE CAU TRUC, khong phai do cau hinh sai. Hai may da phai tu chay
# tay mot lenh bam ben ngoai script moi chung minh duoc.
#
# Gio: SHA cho lich su, bam cho doi chieu cheo may. Ghi ca hai.
# `exit 1` ben trong $(...) chi thoat SUBSHELL nen bam_nguon tra ve ma loi.
kiem_manifest || exit 1
BAM=$(bam_nguon) || {
    echo "DUNG LAI: khong bam duoc ma nguon (thieu SOURCE_MANIFEST, hoac"
    echo "  khong co ca sha256sum lan shasum)."
    exit 1
}
[ -n "$BAM" ] || { echo "DUNG LAI: bam ma nguon ra rong"; exit 1; }

if command -v git > /dev/null 2>&1 && git rev-parse --git-dir > /dev/null 2>&1; then
    # -uno: BO file untracked. `--porcelain` khong co no se liet ke ca `??`, nen
    # bat ky .py nao trong cay lam viec deu chan run -- ke ca venv dat trong repo,
    # dung nhu SERVER.md muc 2 huong dan (22.290 file tren may 3090). Y dinh cua
    # cong la "ma nguon DA COMMIT phai sach", ma file chua track thi khong thuoc
    # ma nguon da commit.
    DIRTY=$(git status --porcelain -uno -- '*.py' '*.sh')
    if [ -n "$DIRTY" ]; then
        echo "DUNG LAI: co thay doi .py/.sh chua commit. Hai may phai cung mot SHA."
        echo "$DIRTY"
        exit 1
    fi
    SHA=$(git rev-parse --short HEAD)
    [ -n "$SHA" ] || { echo "DUNG LAI: khong doc duoc git SHA"; exit 1; }
    XUAT_XU="git=$SHA bam=$BAM"
elif [ "${ALLOW_NO_GIT:-0}" = "1" ]; then
    SHA="(khong-git)"
    XUAT_XU="bam=$BAM"
    echo "  KHONG co git -- chi co bam noi dung, khong co lich su."
else
    echo "DUNG LAI: khong co git."
    echo "  Ba cach, theo thu tu uu tien:"
    echo "    1. sudo apt update && sudo apt install -y git"
    echo "    2. khong co root: cai git vao ./.tools/ (micromamba, ~150 MB)"
    echo "    3. ALLOW_NO_GIT=1 $0 $FOLDS"
    exit 1
fi
echo "  BAM MA NGUON: $BAM   <-- hai may PHAI trung con so nay"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
echo "  SHA $SHA   GPU $GPU   workers $WORKERS"

# Ghi lai xuat xu ra dia. stdout cua tmux se troi mat; file thi con.
{
    echo "xuat_xu=$XUAT_XU"       # git=<sha>  hoac  bam_nguon=<bam>
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
    # Da co run nao cua (TAP, nhanh, fold) nay chay den dong UAR chua?
    #
    # Tien to $DATASET la BAT BUOC. Thu muc log ten
    # log/<TAP>-<timestamp><exper_name>-set<N>-log, ma exper_name khong chua ten
    # tap. Glob khong co tien to se khop thu muc cua TAP KHAC: sau khi DFEW chay
    # xong, moi run MAFW deu bi coi la "da xong" va bo qua -- ca 20 run, in
    # "BO QUA", bao chay 0 run, trong y het thanh cong.
    local ten="$1"
    for d in log/"$DATASET"-*"$ten"-set*-log; do
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
