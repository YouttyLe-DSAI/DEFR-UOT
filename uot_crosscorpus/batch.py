"""Run the cross-corpus OT analysis over every fold and both directions.

Finds the dumps by name, pairs each fold's source with its target, and
aggregates into the mean +- std table the report needs.

  python -m uot_crosscorpus.batch --dumps-root /kaggle/input/mma-baseline-dumps \
      --source-prior target-matched --out uot_results.csv
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import metrics as M
import run as R
import uot as OT

# Both members of a pair must come from the SAME checkpoint, or the features do
# not share a metric space and the cost matrix means nothing.
DIRECTIONS = {
    'dfew2mafw': ('dfew_dfew', 'dfew_mafw11'),
    'mafw2dfew': ('mafw_mafw11', 'mafw_dfew'),
}


def find_dumps(root, case, method):
    """case -> {fold: path}, matching <case>_fold<N>_<method>.npz anywhere under root."""
    found = {}
    for path in glob.glob(os.path.join(root, '**', '*.npz'), recursive=True):
        name = os.path.basename(path)
        if not name.startswith(case + '_'):
            continue
        if method and method not in name:
            continue
        m = re.search(r'fold[_-]?(\d+)', name)
        if m:
            found[int(m.group(1))] = path
    return found


def run_pair(src_path, tgt_path, args):
    n_shared = D.N_SHARED
    X, ys, _, _ = D.load_dump(src_path, args.feat_key, args.label_key)
    Z, yt, tgt_logits, _ = D.load_dump(tgt_path, args.feat_key, args.label_key)

    if args.source_label_space == 'shared':
        keep = ys < n_shared
        X, ys = X[keep], ys[keep]

    a = R.source_prior(ys, yt, n_shared, args.source_prior)
    C = OT.cost_matrix(X, Z, metric=args.metric, normalize=not args.no_normalize)
    plans = R.solve_both(C, args.eps, args.tau, args.iters, a=a)

    out = {}
    for name, pi in plans.items():
        row, _, _ = R.evaluate(pi, ys, yt, n_shared, name)
        out[name] = row

    if tgt_logits is not None and tgt_logits.shape[1] >= n_shared:
        known = yt < n_shared
        zs = tgt_logits[:, :n_shared].argmax(axis=1)
        uar, war, _ = M.uar_war(yt[known], zs[known], n_classes=n_shared)
        out['zero-shot'] = {'method': 'zero-shot', 'uar': uar, 'war': war,
                            'total_mass': float('nan'), 'auroc': float('nan'),
                            'fpr95': float('nan')}
    return out


def aggregate(per_fold, key):
    vals = np.array([r[key] for r in per_fold if np.isfinite(r[key])])
    if len(vals) == 0:
        return float('nan'), float('nan')
    return float(vals.mean()), float(vals.std(ddof=0))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dumps-root', required=True)
    p.add_argument('--direction', default='both',
                   choices=['both'] + sorted(DIRECTIONS))
    p.add_argument('--method', default='av', help='av | visual_only | audio_only')
    p.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--feat-key', default=None)
    p.add_argument('--label-key', default=None)
    p.add_argument('--eps', type=float, default=0.05)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--metric', default='sqeuclidean', choices=['sqeuclidean', 'cosine'])
    p.add_argument('--no-normalize', action='store_true')
    p.add_argument('--source-label-space', default='shared', choices=['shared', 'full'],
                   help="'full' keeps source-only classes (partial adaptation)")
    p.add_argument('--source-prior', default='target-matched',
                   choices=['uniform', 'target-matched'])
    p.add_argument('--out', default=None, help='write per-fold rows to this .csv')
    return p.parse_args()


def main():
    args = parse_args()
    directions = sorted(DIRECTIONS) if args.direction == 'both' else [args.direction]
    csv_rows = ['direction,fold,method,uar,war,total_mass,auroc,fpr95']

    print('dumps root   : {}'.format(args.dumps_root))
    print('source prior : {}'.format(args.source_prior))
    print('eps={}  tau={}  metric={}  method={}'.format(
        args.eps, args.tau, args.metric, args.method))

    for direction in directions:
        src_case, tgt_case = DIRECTIONS[direction]
        srcs = find_dumps(args.dumps_root, src_case, args.method)
        tgts = find_dumps(args.dumps_root, tgt_case, args.method)

        print('\n' + '=' * 68)
        print('{}   source={}  target={}'.format(direction, src_case, tgt_case))
        print('  found {} source dumps, {} target dumps'.format(len(srcs), len(tgts)))
        if not srcs or not tgts:
            print('  !! nothing to pair -- check --dumps-root and --method,')
            print('     expected names like {}_fold1_{}.npz'.format(src_case, args.method))
            continue

        collected = {}
        for fold in args.folds:
            if fold not in srcs or fold not in tgts:
                print('  fold {}: missing dump, skipped'.format(fold))
                continue
            res = run_pair(srcs[fold], tgts[fold], args)
            for name, row in res.items():
                collected.setdefault(name, []).append(row)
                csv_rows.append('{},{},{},{:.4f},{:.4f},{:.6f},{:.4f},{:.4f}'.format(
                    direction, fold, name, row['uar'], row['war'],
                    row['total_mass'], row['auroc'], row['fpr95']))
            print('  fold {}: bal {:.2f} / unbal {:.2f} UAR   (gap {:+.2f})'.format(
                fold, res['balanced']['uar'], res['unbalanced']['uar'],
                res['unbalanced']['uar'] - res['balanced']['uar']))

        if not collected:
            continue
        print('\n  {:<12} {:>16} {:>16} {:>15}'.format('method', 'UAR', 'WAR', 'AUROC'))
        for name in ('zero-shot', 'balanced', 'unbalanced'):
            if name not in collected:
                continue
            um, us = aggregate(collected[name], 'uar')
            wm, ws = aggregate(collected[name], 'war')
            am, asd = aggregate(collected[name], 'auroc')
            print('  {:<12} {:>8.2f} ± {:<5.2f} {:>8.2f} ± {:<5.2f} {:>7.3f} ± {:<5.3f}'.format(
                name, um, us, wm, ws, am, asd))

        if 'balanced' in collected and 'unbalanced' in collected:
            gaps = np.array([u['uar'] - b['uar'] for b, u
                             in zip(collected['balanced'], collected['unbalanced'])])
            n = len(gaps)
            # paired t over folds; with n=5 this is indicative, not conclusive
            t = gaps.mean() / (gaps.std(ddof=1) / np.sqrt(n)) if n > 1 and gaps.std(ddof=1) > 0 else float('nan')
            print('\n  unbalanced - balanced : {:+.2f} UAR   per fold {}'.format(
                gaps.mean(), np.round(gaps, 2).tolist()))
            print('  paired t({}) = {:.2f}'.format(n - 1, t))

    if args.out:
        with open(args.out, 'w') as f:
            f.write('\n'.join(csv_rows) + '\n')
        print('\nwrote {} ({} rows)'.format(args.out, len(csv_rows) - 1))


if __name__ == '__main__':
    main()
