"""Compare two training runs from their log directories.

main.py writes everything needed into log.txt. The line it labels
"Confusion Matrix Diag" is the diagonal of the row-normalised confusion matrix,
i.e. per-class recall, and UAR is its mean -- so the per-class breakdown the
aggregate hides is already there, it just needs reading.

  python tools/compare_runs.py --log-root log --base AB_BASE --uot AB_UOT
"""

import argparse
import ast
import glob
import os
import re

DFEW_CLASSES = ['happiness', 'sadness', 'neutral', 'anger', 'surprise', 'disgust', 'fear']
MAFW_CLASSES = DFEW_CLASSES + ['contempt', 'anxiety', 'helplessness', 'disappointment']


def parse_log(path):
    txt = open(path).read()
    out = {
        'val_acc': [float(x) for x in re.findall(r'Current Accuracy: ([\d.]+)', txt)],
        'epoch_s': [float(x) for x in re.findall(r'An epoch time: ([\d.]+)s', txt)],
        'resumed': bool(re.search(r'resumed_from=', txt)),
        'use_uot': bool(re.search(r'use_uot=True', txt)),
        # use_uot=True dung cho CA BA nhanh attn / tau=1e6 / tau=1.0. Khong doc
        # them hai truong nay thi cong cu khong phan biet duoc chung, va se dan
        # nhan "UOT" cho mot run bat ky. main.py ghi ca vars(args) vao log nen
        # chung co san.
        'uot_mode': (re.search(r'^uot_mode=(\w+)', txt, re.M) or [None, None])[1],
        'uot_tau': (re.search(r'^uot_tau=([\deE.+-]+)', txt, re.M) or [None, None])[1],
        'epochs': None, 'lr': None, 'uar': None, 'war': None, 'recall': None,
    }
    m = re.search(r'^epochs=(\d+)', txt, re.M)
    if m:
        out['epochs'] = int(m.group(1))
    m = re.search(r'^lr=([\d.e+-]+)', txt, re.M)
    if m:
        out['lr'] = m.group(1)

    uar = re.findall(r'UAR: ([\d.]+)', txt)
    war = re.findall(r'WAR: ([\d.]+)', txt)
    if uar:
        out['uar'] = float(uar[-1])
    if war:
        out['war'] = float(war[-1])

    m = re.search(r'\|g\| mean ([\d.]+)\s+max ([\d.]+)', txt)
    out['gate_mean'], out['gate_max'] = (float(m.group(1)), float(m.group(2))) if m else (None, None)

    m = re.search(r'Confusion Matrix Diag:\s*\n(\[[^\]]*\])', txt)
    if m:
        try:
            out['recall'] = ast.literal_eval(m.group(1))
        except (ValueError, SyntaxError):
            pass
    return out


def find_run(log_root, name):
    """Runs matching `name`, completed ones first.

    An interrupted session leaves a log directory with no UAR line. Sorting by
    name alone would put such a directory first and the comparison would report
    'not finished' while a finished run sat right next to it.
    """
    hits = [d for d in glob.glob(os.path.join(log_root, '*'))
            if name in os.path.basename(d) and os.path.isfile(os.path.join(d, 'log.txt'))]
    done = [d for d in hits if parse_log(os.path.join(d, 'log.txt'))['uar'] is not None]
    return sorted(done) + sorted(d for d in hits if d not in done)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--log-root', default='log')
    p.add_argument('--base', default='AB_BASE', help='substring identifying the baseline run')
    p.add_argument('--uot', default='AB_UOT', help='substring identifying the UOT run')
    return p.parse_args()


def show(tag, runs):
    print('\n{}  ({} run)'.format(tag, len(runs)))
    for d, r in runs:
        flags = []
        if r['resumed']:
            flags.append('warm-start')
        if r['use_uot']:
            flags.append('use_uot')
        print('  {}'.format(os.path.basename(d)))
        print('    epochs={}  lr={}  {}'.format(r['epochs'], r['lr'], ' '.join(flags)))
        print('    val acc per epoch : {}'.format([round(a, 2) for a in r['val_acc']]))
        if r['epoch_s']:
            print('    epoch time        : {:.1f} phut'.format(
                sum(r['epoch_s']) / len(r['epoch_s']) / 60))
        print('    UAR {} | WAR {}'.format(r['uar'], r['war']))
        if r['gate_max'] is not None:
            note = '  <- van bang 0: UOT khong dong gop gi' if r['gate_max'] < 1e-3 else ''
            print('    UOT gate |g| mean {:.4f} max {:.4f}{}'.format(
                r['gate_mean'], r['gate_max'], note))


def main():
    args = parse_args()
    base = [(d, parse_log(os.path.join(d, 'log.txt'))) for d in find_run(args.log_root, args.base)]
    uot = [(d, parse_log(os.path.join(d, 'log.txt'))) for d in find_run(args.log_root, args.uot)]

    if not base or not uot:
        print('Chua du hai nhanh: base={}, uot={} duoi {}/'.format(
            len(base), len(uot), args.log_root))
        for d in sorted(glob.glob(os.path.join(args.log_root, '*'))):
            print('  co:', os.path.basename(d))
        return

    show('A. BASELINE', base)
    show('B. UOT', uot)

    n_partial = sum(1 for _, r in base + uot if r['uar'] is None)
    if n_partial:
        print('\n  ({} run bo do khong co dong UAR -- bo qua, dung run da chay xong)'
              .format(n_partial))

    # Ghep cap theo FOLD. Ten thu muc la <DS>-<YYmmddHHMM><exper_name>-set<N>-log,
    # nen sorted() xep theo TIMESTAMP truoc, so fold sau. Lay [0] cua moi ben la
    # lay run som nhat -- chay 5 fold trong mot lenh thi tinh co dung ca hai la
    # fold 1, nhung chay tach phien (dung --folds, dung nhu SERVER.md khuyen) thi
    # se tru hai FOLD KHAC NHAU va in ra "dong gop cua UOT" ma khong he bao gi.
    def fold_of(path):
        m = re.search(r'-set(\d+)-log', os.path.basename(path))
        return int(m.group(1)) if m else None

    base_by_fold = {fold_of(d): r for d, r in base if r['uar'] is not None}
    uot_by_fold = {fold_of(d): r for d, r in uot if r['uar'] is not None}
    chung = sorted(f for f in base_by_fold if f in uot_by_fold and f is not None)
    if not chung:
        print('\nKhong co fold nao chay xong o CA HAI nhanh -- khong the so.')
        print('  baseline co fold: {}'.format(sorted(k for k in base_by_fold if k)))
        print('  nhanh kia co fold: {}'.format(sorted(k for k in uot_by_fold if k)))
        return
    if len(chung) > 1:
        print('\nCo {} fold chung: {}. So tren fold {} (nho nhat).'
              .format(len(chung), chung, chung[0]))
    fold = chung[0]
    a, b = base_by_fold[fold], uot_by_fold[fold]
    print('\nSo tren FOLD {}   |   nhanh B: uot_mode={} uot_tau={}'
          .format(fold, b['uot_mode'], b['uot_tau']))

    print('\n' + '=' * 58)
    print('  {:<12} {:>10} {:>10}'.format('', 'UAR', 'WAR'))
    print('  {:<12} {:>10.2f} {:>10.2f}'.format('A baseline', a['uar'], a['war']))
    print('  {:<12} {:>10.2f} {:>10.2f}'.format('B UOT', b['uar'], b['war']))
    print('  {:<12} {:>+10.2f} {:>+10.2f}   <- dong gop cua UOT'.format(
        'B - A', b['uar'] - a['uar'], b['war'] - a['war']))
    print('=' * 58)

    if a['recall'] and b['recall'] and len(a['recall']) == len(b['recall']):
        names = MAFW_CLASSES if len(a['recall']) == 11 else DFEW_CLASSES
        print('\n  Per-class recall (%) -- UAR la trung binh cot nay')
        print('  {:<15} {:>9} {:>9} {:>9}'.format('class', 'A', 'B', 'B-A'))
        for i, (ra, rb) in enumerate(zip(a['recall'], b['recall'])):
            nm = names[i] if i < len(names) else str(i)
            print('  {:<15} {:>9.2f} {:>9.2f} {:>+9.2f}'.format(nm, ra, rb, rb - ra))
        deltas = [rb - ra for ra, rb in zip(a['recall'], b['recall'])]
        print('\n  bien do lech tung lop: {:+.2f} .. {:+.2f}'.format(min(deltas), max(deltas)))
        print('  Neu cac lop lech manh nhung UAR gan nhu khong doi, thi UAR dang che'
              '\n  mat thay doi that -- bao cao ca bang nay, dung chi bao cao UAR.')

    print('\n  Luu y khi doc: day la 1 fold. Do lech giua cac fold tren bo nay toi'
          '\n  ~12 diem UAR, lon hon nhieu so voi hieu ung thuong thay cua UOT (1-2 diem).'
          '\n  Chi ket luan duoc khi co du 5 fold cho ca hai nhanh.')


if __name__ == '__main__':
    main()
