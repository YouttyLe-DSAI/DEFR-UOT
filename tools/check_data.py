"""Sanity-check an annotation file against the data actually on disk.

Reproduces the exact frame/audio path logic of dataloader/video_dataloader.py
so that a pass here means training will not die at epoch 0 hour 3.

  python tools/check_data.py --annotation annotation/MAFW_set_1_train_faces.txt --n 200
"""

import argparse
import glob
import os
import random


def audio_path_for(frame_path):
    """Mirror of VideoDataset.get() -- returns (wav_path, branch) or (None, 'unmatched')."""
    parts = frame_path.split('/')
    if 'clip_224x224' in frame_path:                       # DFEW
        base = '/'.join(parts[:-2]).replace('clip_224x224', 'raw_wav')
        return os.path.join(base, str(int(parts[-2])) + '.wav'), 'DFEW'
    if 'mfaw' in frame_path:                               # MAFW
        base = '/'.join(parts[:-2]).replace('clips_faces', 'clips_wav')
        return os.path.join(base, parts[-2] + '.wav'), 'MAFW'
    return None, 'unmatched'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--annotation', required=True)
    p.add_argument('--n', type=int, default=100, help='how many entries to sample (0 = all)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rows = [l.strip().rsplit(' ', 2) for l in open(args.annotation) if l.strip()]
    print('{} entries in {}'.format(len(rows), args.annotation))

    sample = rows if args.n == 0 else random.Random(args.seed).sample(rows, min(args.n, len(rows)))

    bad_dir = bad_count = no_wav = unmatched = short = 0
    labels = set()
    for folder, n_frames, label in sample:
        labels.add(int(label))
        if not os.path.isdir(folder):
            bad_dir += 1
            continue
        frames = sorted(glob.glob(os.path.join(folder, '*')))
        if len(frames) != int(n_frames):
            bad_count += 1
        if len(frames) < 16:
            short += 1
        wav, branch = audio_path_for(frames[0] if frames else folder)
        if branch == 'unmatched':
            unmatched += 1
        elif not os.path.exists(wav):
            no_wav += 1

    print('checked        : {}'.format(len(sample)))
    print('missing folder : {}'.format(bad_dir))
    print('frame count off: {}   (fix with tools/retarget_annotations.py --recount)'.format(bad_count))
    print('fewer than 16 frames: {}   (padded by the loader, but worth knowing)'.format(short))
    print('missing .wav   : {}'.format(no_wav))
    print('UNMATCHED path : {}   <-- FATAL: loader hits an undefined `fbank`'.format(unmatched))
    print('labels seen    : {} .. {}  ({} distinct)'.format(min(labels), max(labels), len(labels)))
    if min(labels) != 0:
        print('!! labels do not start at 0 -- CrossEntropyLoss expects 0..C-1')


if __name__ == '__main__':
    main()
