"""The four alignment methods compared in the experiment matrix.

All operate post-hoc on cached features. Nothing is trained: the frozen linear
classifier that shipped with the checkpoint does every classification.

  none              apply the frozen classifier to the target features as they are
  prior_correction  logit adjustment towards an estimated target class prior
  balanced OT       barycentric projection through a balanced transport plan
  unbalanced OT     the same, through a KL-relaxed plan
"""

import numpy as np

import uot as OT

N_SHARED = 7


def classify(Z, clf_w, clf_b, n_classes=N_SHARED):
    """Frozen linear classifier, restricted to the shared label space.

    An MAFW checkpoint emits 11 logits; taking the first 7 rows puts it on the
    same footing as a DFEW checkpoint, since the class order matches.
    """
    W = clf_w[:n_classes]
    b = clf_b[:n_classes]
    return Z @ W.T + b


def estimate_target_prior(logits, src_prior, n_iters=100, tol=1e-8):
    """Saerens-Latinne-Decaestecker EM: recover the target prior from predictions.

    src_prior is the class distribution the classifier was fitted under, taken
    from the labelled source. Substituting the mean prediction on the target for
    it makes the first iteration a no-op and the correction collapses to
    identity, which is what happens if this argument is dropped.

    Uses no target labels, which is what keeps prior correction a baseline rather
    than an oracle.
    """
    p = _softmax(logits)
    src_prior = np.maximum(np.asarray(src_prior, dtype=np.float64), 1e-12)
    prior = src_prior.copy()

    for _ in range(n_iters):
        w = prior / np.maximum(src_prior, 1e-12)
        post = p * w
        post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-12)
        new = post.mean(axis=0)
        if np.max(np.abs(new - prior)) < tol:
            prior = new
            break
        prior = new
    return prior, src_prior


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def prior_correction(logits, src_prior, mode='em', true_target_prior=None):
    """Shift logits by the log ratio of target to source class priors.

    mode='em'     estimate the target prior from the logits themselves
    mode='source' only remove the source prior, assuming a uniform target
    mode='oracle' use the true target prior -- an upper bound, not a baseline
    """
    src_prior = np.maximum(np.asarray(src_prior, dtype=np.float64), 1e-12)

    if mode == 'oracle':
        if true_target_prior is None:
            raise ValueError('oracle prior correction needs the true target prior')
        tgt_prior = true_target_prior
    elif mode == 'source':
        tgt_prior = np.full(logits.shape[1], 1.0 / logits.shape[1])
    else:
        tgt_prior, src_prior = estimate_target_prior(logits, src_prior)

    shift = np.log(np.maximum(tgt_prior, 1e-12)) - np.log(np.maximum(src_prior, 1e-12))
    return logits + shift[None, :], tgt_prior


def barycentric_projection(pi, Z_src, eps=1e-12):
    """Map each target sample to the P-weighted mean of the source features.

    z~_j = sum_i P_ij z_i / sum_i P_ij

    This is what places the target in the source feature space, where the frozen
    classifier is valid. Targets that received no mass -- which unbalanced OT is
    free to produce -- have no meaningful projection; the mask says which.
    """
    mass = pi.sum(axis=0)                       # [n_t]
    projected = (pi.T @ Z_src) / np.maximum(mass, eps)[:, None]
    return projected, mass, mass > eps


def ot_align(Z_src, Z_tgt, clf_w, clf_b, eps=0.05, tau=1.0, n_iters=200,
             balanced=False, a=None, metric='sqeuclidean', normalize=True,
             fallback_logits=None):
    """Transport, project, then classify with the frozen head."""
    C = OT.cost_matrix(Z_src, Z_tgt, metric=metric, normalize=normalize)
    pi = OT.sinkhorn_log(C, a=a, eps=eps,
                         tau=OT.BALANCED_TAU if balanced else tau,
                         n_iters=n_iters)

    projected, mass, ok = barycentric_projection(pi, Z_src)
    logits = classify(projected, clf_w, clf_b)

    # A target that received no mass has an undefined projection. Falling back to
    # its own unaligned logits keeps the comparison honest -- otherwise unbalanced
    # OT would be scored on a subset of the test set.
    if fallback_logits is not None and (~ok).any():
        logits[~ok] = fallback_logits[~ok, :logits.shape[1]]

    return logits, pi, mass, ok
