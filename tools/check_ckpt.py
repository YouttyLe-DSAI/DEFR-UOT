"""Identify what a .pth actually contains before wiring it into the model.

GenerateModel loads its two encoder checkpoints with strict=False, so handing it
the wrong file does not raise -- it quietly loads almost nothing and trains from
near-scratch. This prints enough to tell the three kinds apart:

  vision   : MAE-Face ViT-B encoder    -> mae_face_pretrain_vit_base.pth
  audio    : AudioMAE ViT-B encoder    -> audiomae_pretrained.pth
  trained  : a finished MMA-DFER model -> only useful for evaluate.py

  python tools/check_ckpt.py /kaggle/input/.../pretrained.pth
"""

import argparse
import sys

import torch


def classify(sd):
    keys = list(sd.keys())
    joined = ' '.join(keys)
    if any(k.startswith(('module.', 'our_classifier', 'temporal_net', 'image_encoder', 'audio_model'))
           for k in keys):
        return 'trained'
    if 'decoder_blocks.0.attn.qkv.weight' in joined or 'decoder_pos_embed' in joined:
        return 'mae-with-decoder'
    if 'patch_embed.proj.weight' in joined:
        w = sd['patch_embed.proj.weight']
        return 'audio' if w.shape[1] == 1 else 'vision'
    return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    args = ap.parse_args()

    for path in args.paths:
        print('=' * 70)
        print(path)
        try:
            ckpt = torch.load(path, map_location='cpu')
        except Exception as exc:
            print('  !! cannot load:', exc)
            continue

        if isinstance(ckpt, dict) and 'model' in ckpt:
            sd, wrapper = ckpt['model'], "ckpt['model']"
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd, wrapper = ckpt['state_dict'], "ckpt['state_dict']"
        else:
            sd, wrapper = ckpt, 'raw state_dict'

        print('  top-level keys : {}'.format(
            sorted(k for k in ckpt.keys() if k != 'model')[:8] if isinstance(ckpt, dict) else type(ckpt)))
        print('  weights under  : {}'.format(wrapper))
        if not isinstance(sd, dict):
            print('  !! unexpected structure')
            continue

        kind = classify(sd)
        print('  tensors        : {}'.format(len(sd)))
        print('  params         : {:,}'.format(sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))))
        if 'pos_embed' in sd:
            print('  pos_embed      : {}'.format(tuple(sd['pos_embed'].shape)))
        if 'patch_embed.proj.weight' in sd:
            print('  patch_embed    : {}'.format(tuple(sd['patch_embed.proj.weight'].shape)))
        n_blocks = len({k.split('.')[1] for k in sd if k.startswith('blocks.')})
        if n_blocks:
            print('  blocks         : {}'.format(n_blocks))

        verdict = {
            'vision': "MAE-Face VISION encoder  -> rename to mae_face_pretrain_vit_base.pth",
            'audio': "AudioMAE AUDIO encoder   -> rename to audiomae_pretrained.pth",
            'trained': "a TRAINED MMA-DFER model -> for evaluate.py only, NOT for training",
            'mae-with-decoder': ("MAE with decoder (a 'visualize' checkpoint). The encoder may still\n"
                                 "                   load under strict=False, but prefer the 'pretrain' file."),
            'unknown': "could not classify -- inspect by hand",
        }[kind]
        print('  => {}'.format(verdict))
    print('=' * 70)


if __name__ == '__main__':
    main()
