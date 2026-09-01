"""Pre-resize a frames tree so the dataloader stops paying for it every epoch.

MAFW faces are stored at the resolution they were cropped at, and the loader
decodes them in full and resizes to 224 on every sample, of every epoch, of
every fold. Measured on one worker: 9.6 ms per clip when the source is already
224, 54 ms at 400px, 115 ms at 600px -- an order of magnitude, paid ~757,000
times over a five-fold run.

DFEW already ships at 224 (clip_224x224), so this only brings MAFW in line, and
the team measured 256px against original resolution at t = -0.69, i.e. no
detectable difference in accuracy.

Writes a new tree; the source is left alone. Re-running skips clips that are
already complete; a clip that failed part-way is deleted rather than left short,
so re-running redoes it from scratch. A truncated clip directory is the dangerous
case: nothing downstream can tell it apart from a short video, and --recount would
bless the wrong frame count.

  python tools/resize_frames.py --src /data/tree/mfaw/clips_faces \
      --dst /data/tree224/mfaw/clips_faces --size 224 --jobs 16
"""

import argparse
import os
import shutil
import sys
from multiprocessing import Pool

from PIL import Image

EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def resize_clip(job):
    src_dir, dst_dir, size, quality = job
    os.makedirs(dst_dir, exist_ok=True)
    n_done = n_skip = 0
    for name in os.listdir(src_dir):
        if name.startswith('.') or not name.lower().endswith(EXT):
            continue
        # Always write .jpg: PNG at this size is several times larger for no gain.
        out = os.path.join(dst_dir, os.path.splitext(name)[0] + '.jpg')
        if os.path.exists(out):
            n_skip += 1
            continue
        try:
            with Image.open(os.path.join(src_dir, name)) as im:
                im.convert('RGB').resize((size, size), Image.BILINEAR).save(
                    out, quality=quality)
            n_done += 1
        except Exception as exc:
            # Bao gio cung PHAI xoa thu muc dich khi hong. Truoc day cho nay
            # `return`, tuc thoat ca clip ngay tu anh loi dau tien va de lai mot
            # thu muc CAT NGAN. Dong 38 `if os.path.exists(out): continue` khien
            # chay lai khong tu chua duoc, nen no ton tai vinh vien; roi
            # retarget_annotations.py --recount ghi lai dung so frame bi cat, va
            # check_data.py bao "frame count off: 0". Clip do vinh vien chi duoc
            # lay mau o phan dau video, khong mot canh bao nao.
            shutil.rmtree(dst_dir, ignore_errors=True)
            return (src_dir, 0, 0, str(exc))
    return (src_dir, n_done, n_skip, None)


def dir_size(path):
    total = 0
    for cur, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(cur, f)) for f in files)
    return total


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='directory holding the clip folders')
    p.add_argument('--dst', required=True)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--quality', type=int, default=92)
    p.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    return p.parse_args()


def main():
    args = parse_args()
    clips = sorted(d for d in os.listdir(args.src)
                   if os.path.isdir(os.path.join(args.src, d)) and not d.startswith('.'))
    if not clips:
        sys.exit('no clip folders under ' + args.src)

    print('{} clip folders, {} -> {}, size {}, {} jobs'.format(
        len(clips), args.src, args.dst, args.size, args.jobs))

    jobs = [(os.path.join(args.src, c), os.path.join(args.dst, c), args.size, args.quality)
            for c in clips]
    done = skipped = failed = 0
    with Pool(args.jobs) as pool:
        for i, (clip, n_done, n_skip, err) in enumerate(pool.imap_unordered(resize_clip, jobs), 1):
            done += n_done
            skipped += n_skip
            if err:
                failed += 1
                print('  !! {}: {}'.format(os.path.basename(clip), err))
            if i % 500 == 0 or i == len(jobs):
                print('  {}/{} clip  |  {} anh moi, {} bo qua'.format(i, len(jobs), done, skipped))

    print('\nxong: {} anh ghi moi, {} da co, {} clip loi'.format(done, skipped, failed))
    src_gb, dst_gb = dir_size(args.src) / 1e9, dir_size(args.dst) / 1e9
    print('dung luong: {:.1f} GB -> {:.1f} GB  ({:.1f}x nho hon)'.format(
        src_gb, dst_gb, src_gb / dst_gb if dst_gb else 0))
    if failed:
        # Thoat khac 0 de khong ai chay tiep buoc retarget tren mot cay thieu clip.
        # Cac clip loi da bi xoa han khoi cay dich, nen annexation con tro toi
        # thu muc KHONG ton tai -> dataloader se no ngay, nhin thay duoc. Do la
        # chu y: thieu on con hon cat ngan im lang.
        sys.exit('\nDUNG LAI: {} clip loi, da xoa khoi cay dich. Sua nguyen nhan roi '
                 'chay lai lenh nay truoc khi retarget.'.format(failed))

    print('\nTro annotation sang cay moi:')
    print('  python tools/retarget_annotations.py --dataset MAFW --new-root {} --recount'
          .format(args.dst))


if __name__ == '__main__':
    main()
