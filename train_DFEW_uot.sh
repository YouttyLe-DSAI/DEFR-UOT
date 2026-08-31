#!/bin/bash

CUDA_VISIBLE_DEVICES='0' python main.py \
--dataset 'DFEW' \
--workers 8 \
--epochs 25 \
--batch-size 8 \
--lr 1e-4 \
--weight-decay 1e-2 \
--print-freq 10 \
--milestones 25 \
--temporal-layers 1 \
--img-size 224 \
--use-uot \
--uot-eps 0.05 \
--uot-tau 1.0 \
--uot-iters 10 \
--exper-name UOT_tau1.0_eps0.05 \
