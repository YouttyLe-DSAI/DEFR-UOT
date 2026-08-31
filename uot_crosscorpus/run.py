"""Cross-corpus adaptation between DFEW and MAFW by (unbalanced) optimal transport.

The label spaces are nested -- DFEW's 7 classes are the first 7 of MAFW's 11 --
so transporting DFEW onto MAFW leaves four target classes with no source
counterpart. Balanced OT must still move every unit of target mass somewhere,
which forces those samples onto one of the 7 source classes. Unbalanced OT
relaxes the marginals to a KL penalty and may leave them unmatched.

Both source and target features must come from the SAME checkpoint. Features
from two different encoders do not share a metric, and the cost matrix is then
meaningless -- the reason the audio-visual formulation of this idea failed.

  python -m uot_crosscorpus.run \
      --source Results/baseline_B/dumps/dfew_dfew_fold1_av.npz \
      --target Results/baseline_A/dumps/dfew_mafw11_fold1_av.npz \
      --eps 0.05 --tau 1.0 --sweep
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import metrics as M
import uot as OT


def source_prior(src_labels, tgt_labels, n_shared, mode):
    """Weights for the source marginal `a`.

    'uniform' gives every source sample equal mass, so each class can send only
    as much as its own frequency -- which means balanced OT is penalised by any
    class-prior mismatch, not just by unmatched classes. Since UOT relaxes both
    at once, a sweep run this way cannot attribute its gap to the open-set
    structure alone.

    'target-matched' reweights each source class to the share that class holds
    among the target's *shared-class* samples. That share does not change as the
    sweep adds unmatched samples, so the prior stays fixed while only the
    open-set fraction moves -- which is what H1 actually claims to test.
    """
    n = len(src_labels)
    if mode == 'uniform':
        return np.full(n, 1.0 / n)

    known = tgt_labels[tgt_labels < n_shared]
    if len(known) == 0:
        return np.full(n, 1.0 / n)
    w = np.bincount(known, minlength=n_shared).astype(np.float64)
    w /= w.sum()

    n_cls = int(src_labels.max()) + 1
    counts = np.bincount(src_labels, minlength=n_cls).astype(np.float64)

    # Reweight the shared classes to the target's shared-class proportions, but
    # only within the share of mass they already hold in the source. Any extra
    # source classes keep their own frequency, because how much mass they carry
    # is the partial-adaptation variable under study, not a nuisance to remove.
    shared_share = counts[:n_shared].sum() / counts.sum()

    a = np.zeros(n)
    for c in range(n_cls):
        mask = src_labels == c
        if not mask.any():
            continue
        a[mask] = (shared_share * w[c] / counts[c]) if c < n_shared else (1.0 / counts.sum())
    total = a.sum()
    return a / total if total > 0 else np.full(n, 1.0 / n)


def label_propagate(pi, source_labels, n_classes):
    """Target label = the source class that sent it the most mass."""
    scores = np.zeros((n_classes, pi.shape[1]))
    for c in range(n_classes):
        mask = source_labels == c
        if mask.any():
            scores[c] = pi[mask].sum(axis=0)
    return scores.argmax(axis=0), scores


def evaluate(pi, src_labels, tgt_labels, n_shared, tag):
    # Propagate over whatever label space the source actually has. If extra
    # source classes were kept, a target sample may be assigned one of them --
    # which is simply wrong, and is exactly the error balanced OT is forced into.
    pred, _ = label_propagate(pi, src_labels, int(src_labels.max()) + 1)
    known = tgt_labels < n_shared

    uar, war, _ = M.uar_war(tgt_labels[known], pred[known], n_classes=n_shared)
    mass = pi.sum(axis=0)

    row = {'method': tag, 'uar': uar, 'war': war,
           'total_mass': float(pi.sum()),
           'auroc': float('nan'), 'fpr95': float('nan')}

    if (~known).any():
        # Low received mass should indicate a class the source never saw.
        row['auroc'] = M.auroc(-mass, ~known)
        row['fpr95'] = M.fpr_at_tpr(-mass, ~known)
    return row, pred, mass


def solve_both(C, eps, tau, n_iters, a=None):
    return {
        'balanced': OT.sinkhorn_log(C, a=a, eps=eps, tau=OT.BALANCED_TAU, n_iters=n_iters),
        'unbalanced': OT.sinkhorn_log(C, a=a, eps=eps, tau=tau, n_iters=n_iters),
    }


def print_table(rows, title):
    print('\n' + title)
    print('  {:<12} {:>8} {:>8} {:>11} {:>8} {:>8}'.format(
        'method', 'UAR', 'WAR', 'mass', 'AUROC', 'FPR@95'))
    for r in rows:
        print('  {:<12} {:>8.2f} {:>8.2f} {:>11.4f} {:>8.3f} {:>8.3f}'.format(
            r['method'], r['uar'], r['war'], r['total_mass'], r['auroc'], r['fpr95']))


def sweep(X, Z, tgt_labels, src_labels, n_shared, args, a=None):
    """H1: the balanced/unbalanced gap should grow with the unmatched-class mass."""
    rng = np.random.default_rng(args.seed)
    known_idx = np.flatnonzero(tgt_labels < n_shared)
    novel_idx = np.flatnonzero(tgt_labels >= n_shared)
    if len(novel_idx) == 0:
        print('\n[sweep] target has no unmatched classes -- nothing to sweep')
        return []

    print('\n[sweep] unmatched-class fraction 0 -> 100%  ({} known, {} novel target samples)'
          .format(len(known_idx), len(novel_idx)))
    print('  {:>8} {:>7} {:>10} {:>12} {:>9}'.format(
        'fraction', 'n_novel', 'UAR bal', 'UAR unbal', 'gap'))

    out = []
    for frac in args.sweep_fractions:
        take = int(round(frac * len(novel_idx)))
        keep = np.concatenate([known_idx, rng.choice(novel_idx, take, replace=False)]) \
            if take else known_idx
        keep = np.sort(keep)

        C = OT.cost_matrix(X, Z[keep], metric=args.metric, normalize=not args.no_normalize)
        plans = solve_both(C, args.eps, args.tau, args.iters, a=a)
        res = {}
        for name, pi in plans.items():
            r, _, _ = evaluate(pi, src_labels, tgt_labels[keep], n_shared, name)
            res[name] = r
        gap = res['unbalanced']['uar'] - res['balanced']['uar']
        print('  {:>7.0f}% {:>7d} {:>10.2f} {:>12.2f} {:>+9.2f}'.format(
            frac * 100, take, res['balanced']['uar'], res['unbalanced']['uar'], gap))
        out.append((frac, take, res['balanced']['uar'], res['unbalanced']['uar'], gap))
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help='.npz of labelled source features')
    p.add_argument('--target', required=True, help='.npz of target features (same checkpoint!)')
    p.add_argument('--feat-key', default=None)
    p.add_argument('--label-key', default=None)

    p.add_argument('--eps', type=float, default=0.05, help='entropic regularisation')
    p.add_argument('--tau', type=float, default=1.0, help='marginal relaxation for UOT')
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--metric', default='sqeuclidean', choices=['sqeuclidean', 'cosine'])
    p.add_argument('--no-normalize', action='store_true',
                   help='skip L2 normalisation before building the cost matrix')

    p.add_argument('--target-shared-only', action='store_true',
                   help='drop target samples of the 4 unmatched classes. This is the 0%% '
                        'point of the sweep: with nothing left unmatched, balanced and '
                        'unbalanced OT must agree, and open-set scoring is undefined.')
    p.add_argument('--source-label-space', default='shared', choices=['shared', 'full'],
                   help="'full' keeps source classes absent from the target, which is the "
                        'partial-adaptation setting for MAFW->DFEW. Dropping them removes '
                        'the problem instead of testing it.')
    p.add_argument('--source-prior', default='uniform', choices=['uniform', 'target-matched'],
                   help="'target-matched' removes the class-prior mismatch so the sweep "
                        'measures the open-set effect on its own')
    p.add_argument('--sweep', action='store_true', help='run the H1 fraction sweep')
    p.add_argument('--sweep-fractions', type=float, nargs='+',
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default=None, help='write results as .npz')
    return p.parse_args()


def main():
    args = parse_args()
    n_shared = D.N_SHARED

    print('== dumps ==')
    X, src_labels, src_logits, src_keys = D.load_dump(args.source, args.feat_key, args.label_key)
    D.describe(args.source, X, src_labels, src_keys)
    Z, tgt_labels, tgt_logits, tgt_keys = D.load_dump(args.target, args.feat_key, args.label_key)
    D.describe(args.target, Z, tgt_labels, tgt_keys)

    ck_s, ck_t = D.checkpoint_of(args.source), D.checkpoint_of(args.target)
    if ck_s and ck_t and ck_s != ck_t:
        print('\n  !! source came from the {} checkpoint and target from {}. Their features'
              '\n     do not share a metric space, so the cost matrix is meaningless.'
              '\n     Use two dumps produced by the SAME checkpoint.'.format(ck_s, ck_t))
    if X.shape[1] != Z.shape[1]:
        raise SystemExit('feature dims differ: {} vs {}'.format(X.shape[1], Z.shape[1]))

    keep_src = src_labels < n_shared
    if args.source_label_space == 'shared' and not keep_src.all():
        print('\n  source restricted to the {} shared classes: {} -> {} samples'.format(
            n_shared, len(src_labels), int(keep_src.sum())))
        X, src_labels = X[keep_src], src_labels[keep_src]
    elif not keep_src.all():
        print('\n  source keeps its {} extra classes ({} samples) -- partial adaptation:'
              '\n    balanced OT must spend their mass on the target anyway, unbalanced may not'
              .format(int(src_labels.max()) + 1 - n_shared, int((~keep_src).sum())))

    if args.target_shared_only:
        keep_tgt = tgt_labels < n_shared
        print('  target restricted to shared classes: {} -> {} samples'.format(
            len(tgt_labels), int(keep_tgt.sum())))
        Z, tgt_labels = Z[keep_tgt], tgt_labels[keep_tgt]
        if tgt_logits is not None:
            tgt_logits = tgt_logits[keep_tgt]

    n_novel = int((tgt_labels >= n_shared).sum())
    print('\n== problem ==')
    print('  source {} x {}   target {} x {}   novel target samples: {} ({:.1f}%)'.format(
        X.shape[0], X.shape[1], Z.shape[0], Z.shape[1], n_novel,
        100.0 * n_novel / max(len(tgt_labels), 1)))
    print('  eps={}  tau={}  metric={}  iters={}'.format(
        args.eps, args.tau, args.metric, args.iters))

    a = source_prior(src_labels, tgt_labels, n_shared, args.source_prior)
    print('  source prior: {}'.format(args.source_prior))
    if args.source_prior == 'uniform':
        print('    note: balanced OT is then also penalised by class-prior mismatch, not')
        print('    only by unmatched classes. Use --source-prior target-matched to isolate')
        print('    the open-set effect that H1 is about.')

    C = OT.cost_matrix(X, Z, metric=args.metric, normalize=not args.no_normalize)
    print('  cost: min {:.4f}  mean {:.4f}  max {:.4f}'.format(C.min(), C.mean(), C.max()))

    rows = []
    if tgt_logits is not None:
        known = tgt_labels < n_shared
        zs = tgt_logits[:, :n_shared].argmax(axis=1)
        uar, war, _ = M.uar_war(tgt_labels[known], zs[known], n_classes=n_shared)
        rows.append({'method': 'zero-shot', 'uar': uar, 'war': war, 'total_mass': float('nan'),
                     'auroc': float('nan'), 'fpr95': float('nan')})

    plans = solve_both(C, args.eps, args.tau, args.iters, a=a)
    saved = {}
    for name in ('balanced', 'unbalanced'):
        row, pred, mass = evaluate(plans[name], src_labels, tgt_labels, n_shared, name)
        rows.append(row)
        saved[name + '_pred'] = pred
        saved[name + '_mass'] = mass

    print_table(rows, '== results ==')

    gap = rows[-1]['uar'] - rows[-2]['uar']
    print('\n  unbalanced - balanced = {:+.2f} UAR'.format(gap))
    if n_novel == 0:
        print('  (no unmatched classes present, so the two are expected to coincide)')

    sweep_rows = sweep(X, Z, tgt_labels, src_labels, n_shared, args, a=a) if args.sweep else []

    if args.out:
        np.savez(args.out, rows=np.array([str(r) for r in rows]),
                 sweep=np.array(sweep_rows, dtype=np.float64), **saved)
        print('\nwrote', args.out)


if __name__ == '__main__':
    main()
