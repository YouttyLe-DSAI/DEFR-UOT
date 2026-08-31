"""Run the four alignment methods over both directions and all five folds.

Produces the experiment-matrix numbers: mean over folds for each method, plus
per-class recall, which the aggregate UAR is known to hide.

  python -m uot_crosscorpus.batch --dumps-root dumps --out uot_results.csv
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import run as R

METHODS = ('none', 'prior_correction', 'balanced_ot', 'unbalanced_ot')

# Both members of a pair must come from the SAME checkpoint, or the features
# share no metric space and the cost matrix means nothing.
DIRECTIONS = {
    'dfew2mafw': ('dfew_dfew', 'dfew_mafw11'),
    'mafw2dfew': ('mafw_mafw11', 'mafw_dfew'),
}


def find_dumps(root, case, method):
    found = {}
    for path in glob.glob(os.path.join(root, '**', '*.npz'), recursive=True):
        name = os.path.basename(path)
        if not name.startswith(case + '_') or (method and method not in name):
            continue
        m = re.search(r'fold[_-]?(\d+)', name)
        if m:
            found[int(m.group(1))] = path
    return found


def run_fold(src_path, tgt_path, args):
    Z_s, y_s, _, _ = D.load_dump(src_path, args.feat_key, args.label_key)
    Z_t, y_t, tgt_logits, _ = D.load_dump(tgt_path, args.feat_key, args.label_key)
    clf_w, clf_b = D.load_classifier(tgt_path)

    Z_s, y_s, _, _ = R.restrict_to_shared(Z_s, y_s)
    Z_t, y_t, tgt_logits, _ = R.restrict_to_shared(Z_t, y_t, tgt_logits)
    return R.run_all(Z_s, y_s, Z_t, y_t, tgt_logits, clf_w, clf_b, args)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dumps-root', required=True)
    p.add_argument('--direction', default='both', choices=['both'] + sorted(DIRECTIONS))
    p.add_argument('--method', default='av', help='av | visual_only | audio_only')
    p.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--feat-key', default=None)
    p.add_argument('--label-key', default=None)
    p.add_argument('--eps', type=float, default=0.05)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--metric', default='sqeuclidean', choices=['sqeuclidean', 'cosine'])
    p.add_argument('--no-normalize', action='store_true')
    p.add_argument('--source-prior', default='uniform',
                   choices=['uniform', 'target-matched'])
    p.add_argument('--prior-mode', default='em', choices=['em', 'source', 'oracle'])
    p.add_argument('--out', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    directions = sorted(DIRECTIONS) if args.direction == 'both' else [args.direction]
    csv = ['direction,fold,method,uar,war,' + ','.join('recall_' + c for c in D.DFEW_CLASSES)]

    print('dumps root {}   modality {}   eps={} tau={}   source-prior {}   prior-mode {}'
          .format(args.dumps_root, args.method, args.eps, args.tau,
                  args.source_prior, args.prior_mode))
    print('label space: the {} classes shared by DFEW and MAFW (MAFW-7)'.format(R.N_SHARED))

    for direction in directions:
        src_case, tgt_case = DIRECTIONS[direction]
        srcs = find_dumps(args.dumps_root, src_case, args.method)
        tgts = find_dumps(args.dumps_root, tgt_case, args.method)

        print('\n' + '=' * 72)
        print('{}   source={}  target={}   ({} / {} dumps found)'.format(
            direction, src_case, tgt_case, len(srcs), len(tgts)))
        if not srcs or not tgts:
            print('  !! nothing to pair -- expected names like {}_fold1_{}.npz'.format(
                src_case, args.method))
            continue

        per_method = {m: [] for m in METHODS}
        recalls = {m: [] for m in METHODS}
        for fold in args.folds:
            if fold not in srcs or fold not in tgts:
                print('  fold {}: missing dump, skipped'.format(fold))
                continue
            res = run_fold(srcs[fold], tgts[fold], args)
            for m in METHODS:
                uar, war, rec = res[m]
                per_method[m].append((uar, war))
                recalls[m].append(rec)
                csv.append('{},{},{},{:.4f},{:.4f},{}'.format(
                    direction, fold, m, uar, war,
                    ','.join('{:.4f}'.format(v) for v in rec)))
            print('  fold {}: none {:.2f} | prior {:.2f} | bal {:.2f} | UOT {:.2f}'.format(
                fold, *[res[m][0] for m in METHODS]))

        if not per_method['none']:
            continue

        print('\n  {:<18} {:>16} {:>16}'.format('method', 'UAR', 'WAR'))
        means = {}
        for m in METHODS:
            arr = np.array(per_method[m])
            means[m] = arr[:, 0].mean()
            print('  {:<18} {:>8.2f} ± {:<5.2f} {:>8.2f} ± {:<5.2f}'.format(
                m, arr[:, 0].mean(), arr[:, 0].std(), arr[:, 1].mean(), arr[:, 1].std()))

        print('\n  per-class recall, mean over folds (%)')
        print('  {:<12} {:>9} {:>9} {:>9} {:>9}'.format('class', 'none', 'prior', 'bal OT', 'UOT'))
        for c, name in enumerate(D.DFEW_CLASSES):
            print('  {:<12} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f}'.format(
                name, *[np.array(recalls[m])[:, c].mean() for m in METHODS]))

        best_other = max(means[m] for m in METHODS[:-1])
        margin = means['unbalanced_ot'] - best_other
        gaps = np.array([u[0] - b[0] for b, u
                         in zip(per_method['balanced_ot'], per_method['unbalanced_ot'])])
        n = len(gaps)
        t = (gaps.mean() / (gaps.std(ddof=1) / np.sqrt(n))
             if n > 1 and gaps.std(ddof=1) > 0 else float('nan'))
        print('\n  UOT - best of the other three : {:+.2f} UAR   {}'.format(
            margin, 'MEETS the success criterion' if margin > 0 else 'FAILS the success criterion'))
        print('  UOT - balanced OT, per fold   : {}   paired t({}) = {:.2f}'.format(
            np.round(gaps, 2).tolist(), n - 1, t))

    if args.out:
        with open(args.out, 'w') as f:
            f.write('\n'.join(csv) + '\n')
        print('\nwrote {} ({} rows)'.format(args.out, len(csv) - 1))


if __name__ == '__main__':
    main()
