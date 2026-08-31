"""Load the per-run .npz dumps and reason about the DFEW/MAFW label spaces."""

import os
import re

import numpy as np

# DFEW's 7 classes are the first 7 of MAFW, in the same order, so transporting
# between the two needs no label permutation. The baseline report verified this
# from class frequencies rather than assuming it.
DFEW_CLASSES = ['happiness', 'sadness', 'neutral', 'anger', 'surprise', 'disgust', 'fear']
MAFW_EXTRA = ['contempt', 'anxiety', 'helplessness', 'disappointment']
MAFW_CLASSES = DFEW_CLASSES + MAFW_EXTRA
N_SHARED = len(DFEW_CLASSES)

FEAT_KEYS = ['feature', 'features', 'feat', 'feats', 'embedding', 'embeddings', 'z']
LABEL_KEYS = ['label', 'labels', 'target', 'targets', 'y', 'gt']
LOGIT_KEYS = ['logit', 'logits', 'output', 'outputs', 'pred']


def _pick(npz, candidates, ndim=None):
    for key in candidates:
        if key in npz.files:
            arr = npz[key]
            if ndim is None or arr.ndim == ndim:
                return key, arr
    return None, None


def load_dump(path, feat_key=None, label_key=None):
    """Return (features [N,D], labels [N], logits [N,C] or None, key names used)."""
    npz = np.load(path, allow_pickle=False)

    if feat_key:
        feats = npz[feat_key]
    else:
        feat_key, feats = _pick(npz, FEAT_KEYS, ndim=2)
    if feats is None:
        raise KeyError('no feature array in {}; keys present: {}'.format(path, npz.files))

    if label_key:
        labels = npz[label_key]
    else:
        label_key, labels = _pick(npz, LABEL_KEYS, ndim=1)
    if labels is None:
        raise KeyError('no label array in {}; keys present: {}'.format(path, npz.files))

    _, logits = _pick(npz, LOGIT_KEYS, ndim=2)
    if logits is not None and logits.shape[1] == feats.shape[1]:
        logits = None      # that was the feature array again, not logits

    feats = np.asarray(feats, dtype=np.float64)
    labels = np.asarray(labels).astype(int).ravel()
    if len(feats) != len(labels):
        raise ValueError('{}: {} features vs {} labels'.format(path, len(feats), len(labels)))

    return feats, labels, logits, (feat_key, label_key)


def checkpoint_of(path):
    """Infer which checkpoint produced a dump, from the <ckpt>_<test> case name."""
    name = os.path.basename(path)
    m = re.match(r'(dfew|mafw)_', name.lower())
    return m.group(1) if m else None


def describe(path, feats, labels, keys):
    counts = np.bincount(labels)
    present = [(c, int(n)) for c, n in enumerate(counts) if n]
    print('  {}'.format(os.path.basename(path)))
    print('    features {}  labels {}  (keys: {} / {})'.format(
        feats.shape, labels.shape, keys[0], keys[1]))
    print('    classes  {}  -> {}'.format(
        len(present), ', '.join('{}:{}'.format(c, n) for c, n in present)))
