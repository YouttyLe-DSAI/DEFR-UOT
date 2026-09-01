#!/bin/bash
# ============================================================================
# DAY KHONG PHAI SCRIPT CHAY ABLATION.  Chay ablation:  ./run_ablation.sh <fold...>
#
# Script nay la: MAFW + UOT o 224, MOT nhanh (khong co --uot-mode)
#
# Hai khac biet lam hong ca loat neu chay nham:
#   --img-size 224   ablation chay o 160. O 224 tot gan GAP DOI thoi gian, va
#                    4 nhanh x 5 fold KHONG lot quy thoi gian.
#   thieu --uot-mode  chi ra duoc nhanh 4 (UOT), khong ra duoc nhanh 2 (attn)
#                    va nhanh 3 (OT can bang).
#
# Ca hai deu KHONG bao loi. Dau vet duy nhat la mot dong trong log.txt.
# Giu lai de tai lap cau hinh 224 cua paper, khong phai de chay ablation.
# ============================================================================

CUDA_VISIBLE_DEVICES='0' python main.py \
--dataset 'MAFW' \
--workers 8 \
--epochs 25 \
--batch-size 8 \
--lr 1e-4 \
--weight-decay 1e-2 \
--print-freq 10 \
--temporal-layers 1 \
--img-size 224 \
--use-uot \
--uot-eps 0.05 \
--uot-tau 1.0 \
--uot-iters 10 \
--exper-name UOT_tau1.0_eps0.05 \
