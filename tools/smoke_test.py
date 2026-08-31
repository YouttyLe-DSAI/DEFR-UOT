"""One-batch smoke test: build the model, run forward + backward, verify UOT wiring.

Run this before launching a real job. It costs ~1 minute and catches the
failures that otherwise show up three hours into epoch 0.

  python tools/smoke_test.py --dataset MAFW --use-uot
"""

import argparse
import os
import sys

import torch

# `python tools/smoke_test.py` puts tools/ on sys.path, not the repo root, so the
# repo's own packages are invisible without this. Every other script in tools/
# is self-contained; this one is not.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.video_dataloader import train_data_loader
from models.Generate_Model import GenerateModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='MAFW', choices=['DFEW', 'MAFW'])
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--temporal-layers', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--use-uot', action='store_true')
    p.add_argument('--uot-eps', type=float, default=0.05)
    p.add_argument('--uot-tau', type=float, default=1.0)
    p.add_argument('--uot-iters', type=int, default=10)
    p.add_argument('--uot-detach', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    args.number_class = 11 if args.dataset == 'MAFW' else 7

    if args.dataset == 'MAFW':
        ann = './annotation/MAFW_set_{}_train_faces.txt'.format(args.fold)
    else:
        ann = './annotation/DFEW_set_{}_train.txt'.format(args.fold)

    print('== building model (use_uot={}) =='.format(args.use_uot))
    model = GenerateModel(args=args).cuda()

    n_uot = sum(p.numel() for n, p in model.named_parameters() if 'uot' in n)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('UOT params      : {:,}'.format(n_uot))
    print('total params    : {:,}'.format(sum(p.numel() for p in model.parameters())))

    print('== loading one batch from {} =='.format(ann))
    data = train_data_loader(list_file=ann, num_segments=16, duration=1,
                             image_size=args.img_size, args=args)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                         shuffle=True, num_workers=2, drop_last=True)
    images, target, audio = next(iter(loader))
    print('images {}  audio {}  target {}'.format(tuple(images.shape), tuple(audio.shape), tuple(target.shape)))

    # Dead-audio check. When the .wav is missing the MAFW branch substitutes
    # torch.zeros(512,128), which the normalisation then turns into a CONSTANT
    # (~0.467), not into zeros -- so the giveaway is zero variance, and nothing
    # else in the pipeline complains. Training would run to completion with the
    # audio modality switched off, making any UOT comparison meaningless.
    stds = audio.flatten(1).std(dim=1)
    n_dead = int((stds == 0).sum())
    print('audio liveness : per-sample std {} -> {}/{} dead'.format(
        [round(float(x), 4) for x in stds], n_dead, len(stds)))
    if n_dead == len(stds):
        print('  !! FAIL: every sample has constant audio. The .wav files are not being\n'
              '     found -- check the path logic with tools/check_data.py. Do NOT train.')
    elif n_dead:
        print('  !! {} sample(s) with constant audio; fine if rare, alarming if common.'.format(n_dead))

    images, target, audio = images.cuda(), target.cuda(), audio.cuda()

    print('== forward ==')
    model.train()
    out = model(images, audio)
    print('output {}  finite={}'.format(tuple(out.shape), bool(torch.isfinite(out).all())))

    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    print('loss {:.4f}'.format(loss.item()))

    if args.use_uot:
        # Must run BEFORE the optimizer steps below, which mutate the weights.
        print('== zero-init check: UOT output must equal the baseline at init ==')
        model.eval()
        with torch.no_grad():
            model.use_uot = False
            base = model(images, audio)
            model.use_uot = True
            uot = model(images, audio)
        delta = (base - uot).abs().max().item()
        print('  max|baseline - uot| = {:.3e}  {}'.format(
            delta, 'OK' if delta < 1e-4 else '<-- gates are not zero-initialized'))
        model.train()

        # Everything here sits behind TWO multiplicative zero-init gates: mine
        # (gate_a2v / gate_v2a) and MMA-DFER's own all_gate, which the baseline
        # also initialises to zeros. With all_gate == 0 the whole adapter branch
        # is multiplied away, so on step 0 *every* UOT gradient is exactly zero
        # by construction -- that is not evidence of a disconnected module.
        # all_gate does receive gradient on step 0, so it moves first and the UOT
        # gates start receiving gradient from step 1. Take a few optimizer steps
        # and check they actually come alive.
        print('== UOT gradient check (needs a few steps: see note below) ==')
        uot_gates = [(n, p) for n, p in model.named_parameters()
                     if 'uot' in n and 'gate' in n]
        if not uot_gates:
            print('  !! no UOT parameters found -- is use_uot wired into GenerateModel?')
        else:
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            grads = []
            for step in range(3):
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(images, audio), target)
                loss.backward()
                g = max(float(p.grad.abs().max()) for _, p in uot_gates if p.grad is not None)
                grads.append(g)
                print('  step {}: loss {:.4f}   max |grad| over {} UOT gates = {:.3e}'.format(
                    step, loss.item(), len(uot_gates), g))
                opt.step()

            if max(grads) > 0:
                print('  OK: UOT gates receive gradient once all_gate moves off zero.')
            else:
                print('  !! FAIL: UOT gates never receive gradient -- the module is disconnected.')
            print('  (step 0 being exactly 0.0 is expected: the baseline\'s own all_gate is')
            print('   zero-initialised, so it zeroes the whole adapter branch on the first step.)')

            model.zero_grad(set_to_none=True)

    print('== peak GPU memory: {:.2f} GB =='.format(torch.cuda.max_memory_allocated() / 1024 ** 3))
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
