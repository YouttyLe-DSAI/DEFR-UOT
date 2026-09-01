"""Rewrite the dataset root inside the annotation .txt files.

The shipped annotations point at the authors' cluster
(/scratch/chumache/dfer_datasets/...). Point them at your own copy instead.

  python tools/retarget_annotations.py --dataset DFEW \
      --new-root /data/dfer/dfew/clip_224x224 --recount

Each line is "<frames_dir> <num_frames> <label>". --recount replaces the frame
count with the real number of files on disk, which you want if your
preprocessing produced a different sampling than the original release.
"""

import argparse
import glob
import os
import sys

OLD_ROOTS = {
    'DFEW': '/scratch/chumache/dfer_datasets/dfew/clip_224x224',
    'MAFW': '/scratch/chumache/dfer_datasets/mfaw/clips_faces',
}
FILES = {
    'DFEW': ['DFEW_set_{}_train.txt', 'DFEW_set_{}_test.txt'],
    'MAFW': ['MAFW_set_{}_train_faces.txt', 'MAFW_set_{}_test_faces.txt'],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['DFEW', 'MAFW'])
    p.add_argument('--new-root', required=True,
                   help='directory that directly contains the per-clip frame folders')
    p.add_argument('--old-root', default=None)
    p.add_argument('--annotation-dir', default='./annotation')
    p.add_argument('--recount', action='store_true',
                   help='re-derive num_frames by listing each folder')
    p.add_argument('--drop-missing', action='store_true',
                   help='remove entries whose frame folder is missing OR empty')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    old_root = (args.old_root or OLD_ROOTS[args.dataset]).rstrip('/')
    new_root = args.new_root.rstrip('/')

    if args.dataset == 'MAFW' and 'mfaw' not in new_root:
        print('!! WARNING: dataloader/video_dataloader.py selects the MAFW audio branch\n'
              '   with `elif "mfaw" in path`. Your new root does not contain "mfaw",\n'
              '   so audio loading will crash. Rename the directory or patch the loader.')
    if args.dataset == 'MAFW':
        # Duong dan audio duoc SUY RA tu duong dan anh bang thay chuoi
        # clips_faces -> clips_wav. Mot cay 224 moi resize chi co anh, khong co
        # wav; luc do video_dataloader.py gan fbank = torch.zeros(512,128) va di
        # tiep -- KHONG mot canh bao nao. Ca bon nhanh se train tren audio rong,
        # va so lieu trong van hop ly. Chan tai day, truoc khi ton mot gio GPU nao.
        wav_root = new_root.replace('clips_faces', 'clips_wav')
        if wav_root == new_root or not os.path.isdir(wav_root):
            sys.exit(
                'DUNG LAI: khong thay {}\n'
                '  Dataloader suy duong dan audio tu duong dan anh (clips_faces ->\n'
                '  clips_wav). Thieu thu muc nay thi fbank ra TOAN SO 0, im lang,\n'
                '  va ca bon nhanh mat hoan toan nhanh audio.\n'
                '  Sua: ln -s <cay_goc>/mfaw/clips_wav {}'.format(wav_root, wav_root))
    if args.dataset == 'DFEW' and 'clip_224x224' not in new_root:
        print('!! WARNING: the DFEW audio branch is selected by `if "clip_224x224" in path`.\n'
              '   Your new root does not contain "clip_224x224".')

    for template in FILES[args.dataset]:
        for fold in range(1, 6):
            path = os.path.join(args.annotation_dir, template.format(fold))
            if not os.path.exists(path):
                print('skip (not found):', path)
                continue

            out, missing, changed, empty = [], 0, 0, 0
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                folder, n_frames, label = line.rsplit(' ', 2)
                new_folder = folder.replace(old_root, new_root)
                if new_folder != folder:
                    changed += 1

                if not os.path.isdir(new_folder):
                    missing += 1
                    if args.drop_missing:
                        continue
                else:
                    found = len(glob.glob(os.path.join(new_folder, '*')))
                    if found == 0:
                        # An empty folder passes isdir() but crashes the loader:
                        # _get_train_indices pads an empty array with mode='edge'.
                        empty += 1
                        if args.drop_missing:
                            continue
                    if args.recount:
                        n_frames = str(found)

                out.append('{} {} {}'.format(new_folder, n_frames, label))

            print('{}: {} lines, {} retargeted, {} missing, {} empty{}'.format(
                os.path.basename(path), len(out), changed, missing, empty,
                '' if args.drop_missing or not (missing or empty)
                else '   <-- rerun with --drop-missing'))
            if not args.dry_run:
                with open(path, 'w') as f:
                    f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
