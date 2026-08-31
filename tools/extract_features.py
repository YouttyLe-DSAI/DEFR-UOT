"""Dump the 512-d representation, logits and labels for one checkpoint x test set.

This is the only GPU step the cross-corpus OT analysis needs. Everything after
it runs on CPU from the .npz files this writes.

The feature taken is the input to our_classifier -- i.e. temporal_net's output,
the fused audio-visual representation just before classification. Captured with
a forward hook rather than by editing the model, so the baseline code stays as
it is.

Cross-corpus is the point: --ckpt-dataset selects the checkpoint (and therefore
the number of output classes), --test-dataset selects the data to run it over.
They are meant to differ.

  python tools/extract_features.py \
      --ckpt-dataset DFEW --test-dataset MAFW --fold 1 \
      --checkpoint .../DFEW_224/fold1_224.pth --out-dir dumps/
"""

import argparse
import os
import sys

import numpy as np
import torch
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.video_dataloader import test_data_loader
from models.Generate_Model import GenerateModel

N_CLASSES = {'DFEW': 7, 'MAFW': 11}
# The constant the dataloader's normalisation turns torch.zeros(512,128) into.
DEAD_AUDIO_VALUE = 4.2677393 / (4.5689974 * 2)


def annotation_for(dataset, fold):
    if dataset == 'DFEW':
        return './annotation/DFEW_set_{}_test.txt'.format(fold)
    return './annotation/MAFW_set_{}_test_faces.txt'.format(fold)


def case_name(ckpt_dataset, test_dataset):
    """Match the naming the analysis scripts expect: <ckpt>_<test>."""
    test_tag = 'dfew' if test_dataset == 'DFEW' else 'mafw11'
    return '{}_{}'.format(ckpt_dataset.lower(), test_tag)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt-dataset', required=True, choices=['DFEW', 'MAFW'],
                   help='which corpus the checkpoint was trained on; sets the class count')
    p.add_argument('--test-dataset', required=True, choices=['DFEW', 'MAFW'],
                   help='which corpus to run over; differs from --ckpt-dataset for cross-corpus')
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out-dir', default='dumps')
    p.add_argument('--method', default='av', choices=['av', 'visual_only', 'audio_only'])
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--temporal-layers', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    args.number_class = N_CLASSES[args.ckpt_dataset]

    ann = annotation_for(args.test_dataset, args.fold)
    print('checkpoint : {} ({} classes)'.format(args.checkpoint, args.number_class))
    print('test set   : {}'.format(ann))
    print('method     : {}'.format(args.method))

    # No DataParallel here. It replicates the module, and a forward hook
    # registered on the original then never fires on the replicas -- so on a
    # 2-GPU Kaggle session the features would come back empty or stale. The
    # released checkpoints were saved through DataParallel, so strip the prefix
    # and load into the bare model instead.
    model = GenerateModel(args=args).cuda()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = state.get('state_dict', state)
    state = {(k[len('module.'):] if k.startswith('module.') else k): v
             for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # our_classifier's input is the 512-d fused representation.
    grabbed = {}

    def hook(_module, inputs, _output):
        grabbed['z'] = inputs[0].detach().float().cpu()

    handle = model.our_classifier.register_forward_hook(hook)

    data = test_data_loader(list_file=ann, num_segments=16, duration=1,
                            image_size=args.img_size)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

    feats, logits, labels = [], [], []
    with torch.no_grad():
        for images, target, audio in tqdm.tqdm(loader):
            images, audio = images.cuda(), audio.cuda()
            if args.method == 'visual_only':
                audio = torch.full_like(audio, DEAD_AUDIO_VALUE)
            elif args.method == 'audio_only':
                images = torch.zeros_like(images)

            out = model(images, audio)
            feats.append(grabbed['z'].numpy())
            logits.append(out.detach().float().cpu().numpy())
            labels.append(target.numpy())

    handle.remove()

    feats = np.concatenate(feats)
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)

    os.makedirs(args.out_dir, exist_ok=True)
    name = '{}_fold{}_{}.npz'.format(case_name(args.ckpt_dataset, args.test_dataset),
                                     args.fold, args.method)
    path = os.path.join(args.out_dir, name)
    np.savez_compressed(path, feature=feats, logit=logits, label=labels)

    acc = 100.0 * (logits.argmax(1) == labels).mean()
    print('\nwrote {}'.format(path))
    print('  feature {}  logit {}  label {}'.format(feats.shape, logits.shape, labels.shape))
    print('  classes seen: {}'.format(sorted(set(labels.tolist()))))
    print('  top-1 over the raw logits: {:.2f}%  (low is expected when the checkpoint'
          '\n  and the test corpus differ, or when a class is outside its output space)'.format(acc))


if __name__ == '__main__':
    main()
