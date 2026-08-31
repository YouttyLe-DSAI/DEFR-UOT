"""Build a symlink tree on Kaggle that matches the layout MMA-DFER's dataloader expects.

The loader derives the .wav path from the frame path by string substitution
(`clips_faces`->`clips_wav`, `clip_224x224`->`raw_wav`) and picks its branch by
substring (`mfaw`, `clip_224x224`). On Kaggle the frames live in several
read-only mounts with different names, and the audio lives in yet another
mount, so that substitution cannot possibly work.

Rather than patch the baseline loader, we present it the layout it wants:

    <out>/mfaw/clips_faces/<clip_id>       -> symlink to whichever shard has it
    <out>/mfaw/clips_wav/<clip_id>.wav     -> symlink into the audio mount
    <out>/dfew/clip_224x224/<clip_id>      -> symlink
    <out>/dfew/raw_wav/<int(clip_id)>.wav  -> symlink, leading zeros stripped

Symlinks cost no disk, so put the tree in /kaggle/temp and rebuild it each session.

Usage:
    python tools/kaggle_setup.py --diagnose
    python tools/kaggle_setup.py --dataset MAFW \
        --frames /kaggle/input/mafw-faces-native-part1/mafw_faces_native_shard0 \
                 /kaggle/input/mafw-faces-native-part2/mafw_faces_native_shard1 \
        --audio  /kaggle/input/mafw-native-rate-audio-for-mma-dfer/mfaw/clips_wav
"""

import argparse
import glob
import os
import re
import sys

JUNK = re.compile(r'(^\._|^\.DS_Store$|^__MACOSX$)')


def is_junk(name):
    return bool(JUNK.match(name))


def diagnose(root='/kaggle/input', max_depth=3):
    print('=' * 70)
    print('MOUNTED INPUT DATASETS under', root)
    print('=' * 70)
    if not os.path.isdir(root):
        print('  (not found -- are you running outside Kaggle?)')
        return
    for slug in sorted(os.listdir(root)):
        print('\n[{}]'.format(slug))
        base = os.path.join(root, slug)
        for cur, dirs, files in os.walk(base):
            depth = cur[len(base):].count(os.sep)
            if depth >= max_depth:
                dirs[:] = []
                continue
            dirs[:] = sorted(d for d in dirs if not is_junk(d))
            rel = cur[len(base):] or '/'
            files = [f for f in files if not is_junk(f)]
            note = ''
            if dirs and all(d.isdigit() for d in dirs[:5]):
                note = '   <-- looks like CLIP FOLDERS ({} of them)'.format(len(dirs))
            if files and files[0].lower().endswith('.wav'):
                note = '   <-- looks like AUDIO ({} wav)'.format(len(files))
            print('  {:<45} dirs={:<6} files={}{}'.format(rel, len(dirs), len(files), note))
            junk = [f for f in os.listdir(cur) if is_junk(f)] if depth < 2 else []
            if junk:
                print('    note: {} macOS junk entr(ies) e.g. {} -- ignored by glob(), harmless'.format(
                          len(junk), junk[0]))


def index_clips(frame_roots):
    """clip_id -> source directory, first root wins."""
    index = {}
    for root in frame_roots:
        if not os.path.isdir(root):
            print('  !! frames root not found:', root)
            continue
        n = 0
        for name in os.listdir(root):
            if is_junk(name):
                continue
            path = os.path.join(root, name)
            if os.path.isdir(path) and name not in index:
                index[name] = path
                n += 1
        print('  {:<70} {} clips'.format(root, n))
    return index


def index_wavs(audio_roots):
    """Normalized key (int if numeric) -> wav path, so 00123.wav and 123.wav both match."""
    index = {}
    for root in audio_roots:
        if not os.path.isdir(root):
            print('  !! audio root not found:', root)
            continue
        wavs = [w for w in glob.glob(os.path.join(root, '**', '*.wav'), recursive=True)
                if not is_junk(os.path.basename(w))]
        for w in wavs:
            stem = os.path.splitext(os.path.basename(w))[0]
            index[int(stem) if stem.isdigit() else stem] = w
        print('  {:<70} {} wav'.format(root, len(wavs)))
    return index


def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def build(args):
    if args.dataset == 'MAFW':
        frames_dir = os.path.join(args.out, 'mfaw', 'clips_faces')
        audio_dir = os.path.join(args.out, 'mfaw', 'clips_wav')
    else:
        frames_dir = os.path.join(args.out, 'dfew', 'clip_224x224')
        audio_dir = os.path.join(args.out, 'dfew', 'raw_wav')

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    print('\n-- indexing frames --')
    clips = index_clips(args.frames)
    print('-- indexing audio --')
    wavs = index_wavs(args.audio)
    print('\ntotal unique clips: {}   total wav: {}'.format(len(clips), len(wavs)))

    n_frame_links = n_wav_links = n_wav_missing = 0
    for clip_id, src in clips.items():
        link(src, os.path.join(frames_dir, clip_id))
        n_frame_links += 1

        key = int(clip_id) if clip_id.isdigit() else clip_id
        wav = wavs.get(key)
        if wav is None:
            n_wav_missing += 1
            continue
        # DFEW: loader asks for str(int(id)).wav   MAFW: loader asks for id.wav
        wav_name = (str(int(clip_id)) if args.dataset == 'DFEW' else clip_id) + '.wav'
        link(wav, os.path.join(audio_dir, wav_name))
        n_wav_links += 1

    print('\n-- built {} --'.format(args.out))
    print('  frame symlinks : {}'.format(n_frame_links))
    print('  wav symlinks   : {}'.format(n_wav_links))
    print('  clips w/o wav  : {}{}'.format(
        n_wav_missing,
        '   <-- FATAL for DFEW (no fallback in the loader)' if args.dataset == 'DFEW' and n_wav_missing
        else '   (MAFW loader substitutes zeros -- see warning below)' if n_wav_missing else ''))

    if n_wav_missing and args.dataset == 'MAFW':
        print('\n  !! WARNING: the MAFW branch silently replaces a missing .wav with\n'
              '     torch.zeros(512,128). If many clips are missing audio, training will\n'
              '     appear to work while the audio modality is dead -- which makes the\n'
              '     whole UOT comparison meaningless. Check this number is small.')

    verify(frames_dir, args.dataset)
    print('\nNEXT: point the annotations at this tree:')
    print('  python tools/retarget_annotations.py --dataset {} --new-root {} --recount'.format(
        args.dataset, frames_dir))


def verify(frames_dir, dataset):
    print('\n-- verifying against the dataloader\'s own path logic --')
    sample = sorted(d for d in os.listdir(frames_dir) if not is_junk(d))[:3]
    if not sample:
        print('  !! no clips linked')
        return
    ok = True
    for clip_id in sample:
        files = sorted(glob.glob(os.path.join(frames_dir, clip_id, '*')))  # skips dotfiles
        junk = [f for f in os.listdir(os.path.join(frames_dir, clip_id)) if is_junk(f)]
        if not files:
            print('  {}: EMPTY'.format(clip_id))
            ok = False
            continue
        p = files[0]
        if 'clip_224x224' in p:
            wav = '/'.join(p.split('/')[:-2]).replace('clip_224x224', 'raw_wav') \
                  + '/' + str(int(p.split('/')[-2])) + '.wav'
            br = 'DFEW'
        elif 'mfaw' in p:
            wav = '/'.join(p.split('/')[:-2]).replace('clips_faces', 'clips_wav') \
                  + '/' + p.split('/')[-2] + '.wav'
            br = 'MAFW'
        else:
            print('  {}: NO BRANCH MATCHES -> UnboundLocalError at train time'.format(clip_id))
            ok = False
            continue
        exists = os.path.exists(wav)
        ok = ok and exists and br == dataset
        print('  {}: {} frames, branch={}, wav {} {}'.format(
            clip_id, len(files), br, 'OK' if exists else 'MISSING', wav))
        if junk:
            # glob('*') skips dotfiles, and the dataloader uses the same glob,
            # so these are inert -- they only inflate `du`.
            print('     note: {} macOS junk file(s) present (e.g. {}); harmless, '
                  'glob() ignores dotfiles'.format(len(junk), os.path.basename(junk[0])))
    print('  => {}'.format('LAYOUT OK' if ok else 'LAYOUT BROKEN -- fix before training'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--diagnose', action='store_true', help='just print what is mounted')
    p.add_argument('--dataset', choices=['MAFW', 'DFEW'])
    p.add_argument('--frames', nargs='+', default=[], help='one or more shard directories')
    p.add_argument('--audio', nargs='+', default=[], help='directory/directories holding the .wav')
    p.add_argument('--out', default='/kaggle/temp/data')
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    if a.diagnose or not a.dataset:
        diagnose()
        if not a.dataset:
            print('\nNow re-run with --dataset MAFW --frames <...> --audio <...>')
        sys.exit(0)
    build(a)
