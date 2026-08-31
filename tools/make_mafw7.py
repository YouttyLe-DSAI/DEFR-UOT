"""Write MAFW-7: the MAFW split restricted to the 7 classes DFEW also has.

DFEW's 7 classes are the first 7 of MAFW's 11, in the same index order, so a
DFEW-trained model (7 outputs) can be scored on these files directly with no
label remapping. Lines whose label is >= 7 are dropped.

  python tools/make_mafw7.py --split test
"""

import argparse
import os
from collections import Counter

N_SHARED = 7
CLASSES = ['happiness', 'sadness', 'neutral', 'anger', 'surprise', 'disgust', 'fear']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--annotation-dir', default='./annotation')
    p.add_argument('--split', default='test', choices=['test', 'train', 'both'])
    p.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    return p.parse_args()


def main():
    args = parse_args()
    splits = ['train', 'test'] if args.split == 'both' else [args.split]

    for split in splits:
        for fold in args.folds:
            src = os.path.join(args.annotation_dir,
                               'MAFW_set_{}_{}_faces.txt'.format(fold, split))
            if not os.path.exists(src):
                print('skip (not found):', src)
                continue

            kept, dropped, counts = [], 0, Counter()
            for line in open(src):
                line = line.strip()
                if not line:
                    continue
                label = int(line.rsplit(' ', 1)[1])
                if label < N_SHARED:
                    kept.append(line)
                    counts[label] += 1
                else:
                    dropped += 1

            dst = src.replace('.txt', '7.txt')
            with open(dst, 'w') as f:
                f.write('\n'.join(kept) + '\n')
            print('{}  ->  {} dong giu, {} dong bo (lop >= {})'.format(
                os.path.basename(dst), len(kept), dropped, N_SHARED))
            if fold == args.folds[0]:
                print('   phan bo lop: ' + ', '.join(
                    '{}={}'.format(CLASSES[c], counts[c]) for c in sorted(counts)))
                print('   Doi chieu thu tu lop voi DFEW truoc khi tin ket qua:'
                      ' index 5 phai la disgust o CA HAI bo.')


if __name__ == '__main__':
    main()
