"""Post-hoc cross-corpus alignment for DFER, on frozen features.

Everything here works in the 7-class label space shared by DFEW and MAFW.
MAFW is restricted to those 7 classes (MAFW-7); the four MAFW-only classes are
dropped, and an MAFW checkpoint's 11 logits are cut to their first 7, which the
matching class order makes valid. Both checkpoints are then compared under
identical conditions.

Four methods, none of which updates a parameter:

  none              frozen classifier on the target features as they are
  prior_correction  logit adjustment towards an EM-estimated target prior
  balanced_ot       barycentric projection through a balanced plan, then classify
  unbalanced_ot     the same through a KL-relaxed plan

  python -m uot_crosscorpus.run \
      --source dumps/dfew_dfew_fold1_av.npz \
      --target dumps/dfew_mafw11_fold1_av.npz \
      --eps 0.05 --tau 1.0
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import methods as ME
import metrics as M
import uot as OT

N_SHARED = ME.N_SHARED


def restrict_to_shared(Z, y, logits=None):
    """Keep only the 7 classes both corpora share. This is what makes MAFW-7."""
    keep = y < N_SHARED
    return Z[keep], y[keep], (logits[keep] if logits is not None else None), keep


def source_prior_weights(y_src, y_tgt, mode):
    """Weights for the OT source marginal `a`.

    'uniform' is the default and the setting the experiment is about: class-prior
    shift is the phenomenon under study, so it must reach the solver intact.
    Balanced OT then has to match both marginals exactly and pushes mass onto the
    over-represented class, which is precisely the failure unbalanced OT avoids.

    'target-matched' hand-corrects that mismatch before the solver sees it. It
    answers a different question -- what transport buys once priors already agree
    -- and removes most of the reason to prefer the unbalanced solver.
    """
    n = len(y_src)
    if mode == 'uniform':
        return np.full(n, 1.0 / n)

    w = np.bincount(y_tgt, minlength=N_SHARED).astype(np.float64)
    w /= max(w.sum(), 1e-12)
    counts = np.bincount(y_src, minlength=N_SHARED).astype(np.float64)

    a = np.zeros(n)
    for c in range(N_SHARED):
        mask = y_src == c
        if mask.any() and counts[c] > 0:
            a[mask] = w[c] / counts[c]
    total = a.sum()
    return a / total if total > 0 else np.full(n, 1.0 / n)


def run_all(Z_s, y_s, Z_t, y_t, tgt_logits, clf_w, clf_b, args):
    """Returns {method: (uar, war, per_class_recall)}."""
    base_logits = ME.classify(Z_t, clf_w, clf_b)
    a = source_prior_weights(y_s, y_t, args.source_prior)
    out = {}

    def score(name, logits):
        pred = logits.argmax(axis=1)
        uar, war, rec = M.uar_war(y_t, pred, n_classes=N_SHARED)
        out[name] = (uar, war, rec)

    score('none', base_logits)

    src_prior = np.bincount(y_s, minlength=N_SHARED).astype(np.float64)
    src_prior /= src_prior.sum()
    adjusted, est_prior = ME.prior_correction(base_logits, src_prior, mode=args.prior_mode)
    score('prior_correction', adjusted)

    for name, balanced in (('balanced_ot', True), ('unbalanced_ot', False)):
        logits, _, mass, ok = ME.ot_align(
            Z_s, Z_t, clf_w, clf_b, eps=args.eps, tau=args.tau,
            n_iters=args.iters, balanced=balanced, a=a, metric=args.metric,
            normalize=not args.no_normalize, fallback_logits=base_logits)
        score(name, logits)
        if not balanced:
            out['_unmatched'] = int((~ok).sum())

    out['_est_prior'] = est_prior
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--feat-key', default=None)
    p.add_argument('--label-key', default=None)
    p.add_argument('--eps', type=float, default=0.05)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--metric', default='sqeuclidean', choices=['sqeuclidean', 'cosine'])
    p.add_argument('--no-normalize', action='store_true')
    p.add_argument('--source-prior', default='uniform',
                   choices=['uniform', 'target-matched'],
                   help="keep 'uniform': prior shift is the phenomenon under study and "
                        "must reach the solver. 'target-matched' corrects it beforehand "
                        'and answers a different question.')
    p.add_argument('--prior-mode', default='em', choices=['em', 'source', 'oracle'],
                   help="'em' estimates the target prior without labels; 'oracle' uses "
                        'the true one and is an upper bound, not a baseline')
    return p.parse_args()


def main():
    args = parse_args()

    print('== dumps ==')
    Z_s, y_s, _, ks = D.load_dump(args.source, args.feat_key, args.label_key)
    D.describe(args.source, Z_s, y_s, ks)
    Z_t, y_t, tgt_logits, kt = D.load_dump(args.target, args.feat_key, args.label_key)
    D.describe(args.target, Z_t, y_t, kt)

    ck_s, ck_t = D.checkpoint_of(args.source), D.checkpoint_of(args.target)
    if ck_s and ck_t and ck_s != ck_t:
        print('\n  !! source is from the {} checkpoint and target from {}; their features'
              '\n     share no metric space and the cost matrix is meaningless.'.format(ck_s, ck_t))

    clf_w, clf_b = D.load_classifier(args.target)
    print('\n  frozen classifier from the target dump: {} + {}'.format(
        clf_w.shape, clf_b.shape))

    n_s0, n_t0 = len(y_s), len(y_t)
    Z_s, y_s, _, _ = restrict_to_shared(Z_s, y_s)
    Z_t, y_t, tgt_logits, _ = restrict_to_shared(Z_t, y_t, tgt_logits)
    print('  restricted to the {} shared classes: source {} -> {}, target {} -> {}'.format(
        N_SHARED, n_s0, len(y_s), n_t0, len(y_t)))

    print('\n== class prior (%) ==')
    ps = 100.0 * np.bincount(y_s, minlength=N_SHARED) / len(y_s)
    pt = 100.0 * np.bincount(y_t, minlength=N_SHARED) / len(y_t)
    print('  {:<12} {:>8} {:>8} {:>8}'.format('class', 'source', 'target', 'ratio'))
    for c in range(N_SHARED):
        print('  {:<12} {:>8.1f} {:>8.1f} {:>7.2f}x'.format(
            D.DFEW_CLASSES[c], ps[c], pt[c], pt[c] / max(ps[c], 1e-9)))

    res = run_all(Z_s, y_s, Z_t, y_t, tgt_logits, clf_w, clf_b, args)

    print('\n== results (7-class label space) ==')
    print('  {:<18} {:>8} {:>8}'.format('method', 'UAR', 'WAR'))
    for name in ('none', 'prior_correction', 'balanced_ot', 'unbalanced_ot'):
        uar, war, _ = res[name]
        print('  {:<18} {:>8.2f} {:>8.2f}'.format(name, uar, war))

    print('\n== per-class recall (%) ==')
    print('  {:<12} {:>9} {:>9} {:>9} {:>9}'.format(
        'class', 'none', 'prior', 'bal OT', 'UOT'))
    for c in range(N_SHARED):
        print('  {:<12} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f}'.format(
            D.DFEW_CLASSES[c], res['none'][2][c], res['prior_correction'][2][c],
            res['balanced_ot'][2][c], res['unbalanced_ot'][2][c]))

    if '_unmatched' in res and res['_unmatched']:
        print('\n  {} target samples received no transport mass; scored on their own'
              '\n  unaligned logits so the test set stays whole.'.format(res['_unmatched']))

    best_other = max(res[m][0] for m in ('none', 'prior_correction', 'balanced_ot'))
    print('\n  UOT vs best of the other three: {:+.2f} UAR'.format(
        res['unbalanced_ot'][0] - best_other))
    print('  (the success criterion requires UOT to beat all three, prior correction included)')


if __name__ == '__main__':
    main()
