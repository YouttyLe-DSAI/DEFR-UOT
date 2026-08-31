"""Entropic optimal transport, balanced and unbalanced, in the log domain.

NumPy only -- these problems run on CPU in seconds, and avoiding a POT
dependency keeps the analysis reproducible from a bare Kaggle image.

Balanced Sinkhorn is the tau -> inf limit of the unbalanced solver, so both
come from one code path and the comparison between them cannot drift.
"""

import numpy as np

BALANCED_TAU = 1e12   # large enough that tau/(tau+eps) == 1.0 in float64


def _logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)), axis=axis)


def sinkhorn_log(C, a=None, b=None, eps=0.05, tau=BALANCED_TAU, n_iters=200, tol=1e-9):
    """Solve  min <C,pi> + eps*KL(pi|a x b) + tau*KL(pi1|a) + tau*KL(pi^T 1|b).

    Args:
        C: cost matrix [N, M].
        a, b: source/target measures, default uniform. Must be positive.
        eps: entropic regularisation, relative to the scale of C.
        tau: marginal relaxation. BALANCED_TAU recovers classical Sinkhorn,
            where the marginals are matched exactly and no mass may be
            created or destroyed.
        n_iters: maximum scaling iterations.
        tol: stop once the potentials move less than this.

    Returns:
        pi: transport plan [N, M]. Total mass is 1 for the balanced case and
            <= 1 otherwise; the shortfall is exactly the mass the solver chose
            not to transport, which is the signal open-set scoring uses.
    """
    N, M = C.shape
    log_a = np.log(a) if a is not None else np.full(N, -np.log(N))
    log_b = np.log(b) if b is not None else np.full(M, -np.log(M))

    f = np.zeros(N)
    g = np.zeros(M)
    scale = tau / (tau + eps)

    for _ in range(n_iters):
        f_prev = f
        f = -scale * eps * _logsumexp((g[None, :] - C) / eps + log_b[None, :], axis=1)
        g = -scale * eps * _logsumexp((f[:, None] - C) / eps + log_a[:, None], axis=0)
        if np.max(np.abs(f - f_prev)) < tol:
            break

    log_pi = (f[:, None] + g[None, :] - C) / eps + log_a[:, None] + log_b[None, :]
    return np.exp(log_pi)


def cost_matrix(X, Z, metric='sqeuclidean', normalize=True):
    """Pairwise cost between source rows X and target rows Z.

    Both must come from the *same* encoder, otherwise the two point clouds do
    not share a metric and the cost is meaningless -- the failure mode that
    sank the audio-visual formulation.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    if metric == 'cosine':
        return 1.0 - X @ Z.T
    if metric == 'sqeuclidean':
        C = (X ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2.0 * (X @ Z.T)
        return np.maximum(C, 0.0)
    raise ValueError('unknown metric: ' + metric)
