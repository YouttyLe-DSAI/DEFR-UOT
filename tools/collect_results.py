"""Gather every run's metrics.json into one table.

Reads the structured log main.py writes, not log.txt -- regexing a
human-readable log breaks whenever a print statement moves, and the report
table should not depend on that.

  python tools/collect_results.py --log-root log --csv results.csv
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict


def load_runs(log_root):
    runs = []
    for path in sorted(glob.glob(os.path.join(log_root, '*', 'metrics.json'))):
        try:
            m = json.load(open(path))
        except (ValueError, OSError):
            continue
        if not m.get('final', {}).get('uar'):
            continue                     # chua chay xong
        cfg = m.get('config', {})
        name = os.path.basename(os.path.dirname(path))
        exper = cfg.get('exper_name') or (re.search(r'\d{10}(.+?)-set', name) or [None, name])[1]
        runs.append({
            'run': name, 'exper': exper, 'fold': m.get('fold'),
            'dataset': m.get('dataset'), 'use_uot': bool(cfg.get('use_uot')),
            'epochs': cfg.get('epochs'), 'lr': cfg.get('lr'),
            'batch': cfg.get('batch_size'), 'img': cfg.get('img_size'),
            'classes': cfg.get('number_class'),
            'uar': m['final']['uar'], 'war': m['final']['war'],
            'best_val': m['final'].get('best_val_acc'),
            'per_class': m['final'].get('per_class_recall', []),
            'names': m['final'].get('class_names', []),
            'gate': (m['final'].get('uot_gates') or {}).get('max_abs'),
            'epoch_min': (sum(e['epoch_seconds'] for e in m['epochs_log'])
                          / max(len(m['epochs_log']), 1) / 60),
        })
    return runs


def mean_std(xs):
    if not xs:
        return float('nan'), float('nan')
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return mu, var ** 0.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--log-root', default='log')
    p.add_argument('--csv', default=None, help='also write the per-run rows here')
    return p.parse_args()


def main():
    args = parse_args()
    runs = load_runs(args.log_root)
    if not runs:
        print('Khong tim thay metrics.json nao da chay xong duoi {}/'.format(args.log_root))
        print('  (run cu truoc ban va nay chi co log.txt -- dung tools/compare_runs.py)')
        return

    print('== tung run ({}) =='.format(len(runs)))
    print('  {:<26} {:>4} {:>4} {:>7} {:>7} {:>8} {:>8}'.format(
        'exper', 'fold', 'uot', 'UAR', 'WAR', 'gate', 'phut/ep'))
    for r in runs:
        print('  {:<26} {:>4} {:>4} {:>7.2f} {:>7.2f} {:>8} {:>8.1f}'.format(
            (r['exper'] or '')[:26], r['fold'], 'yes' if r['use_uot'] else '-',
            r['uar'], r['war'],
            '{:.4f}'.format(r['gate']) if r['gate'] is not None else '-',
            r['epoch_min']))

    groups = defaultdict(list)
    for r in runs:
        groups[r['exper']].append(r)

    print('\n== trung binh theo fold ==')
    print('  {:<26} {:>6} {:>16} {:>16}'.format('exper', 'n fold', 'UAR', 'WAR'))
    for exper, rs in sorted(groups.items()):
        um, us = mean_std([r['uar'] for r in rs])
        wm, ws = mean_std([r['war'] for r in rs])
        note = ''
        if len(rs) < 5:
            note = '   <- chua du 5 fold, do lech giua fold tren bo nay ~12 diem UAR'
        print('  {:<26} {:>6} {:>8.2f} ± {:<5.2f} {:>8.2f} ± {:<5.2f}{}'.format(
            (exper or '')[:26], len(rs), um, us, wm, ws, note))

    per_class = {e: rs for e, rs in groups.items() if rs[0]['per_class']}
    if per_class:
        names = list(per_class.values())[0][0]['names']
        print('\n== recall tung lop, trung binh theo fold (%) ==')
        print('  {:<14} {}'.format('class', ' '.join('{:>12}'.format(e[:12])
                                                     for e in sorted(per_class))))
        for i, cname in enumerate(names):
            row = []
            for e in sorted(per_class):
                vals = [r['per_class'][i] for r in per_class[e] if i < len(r['per_class'])]
                row.append('{:>12.2f}'.format(mean_std(vals)[0]))
            print('  {:<14} {}'.format(cname, ' '.join(row)))

    if args.csv:
        cols = ['run', 'exper', 'dataset', 'fold', 'use_uot', 'epochs', 'lr',
                'batch', 'img', 'classes', 'uar', 'war', 'best_val', 'gate', 'epoch_min']
        with open(args.csv, 'w') as f:
            f.write(','.join(cols + ['recall_' + n for n in (runs[0]['names'] or [])]) + '\n')
            for r in runs:
                f.write(','.join([str(r[c]) for c in cols]
                                 + ['{:.4f}'.format(v) for v in r['per_class']]) + '\n')
        print('\nwrote ' + args.csv)


if __name__ == '__main__':
    main()
