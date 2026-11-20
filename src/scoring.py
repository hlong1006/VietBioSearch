"""Score normalization, reciprocal rank fusion and adaptive result counts."""

import math
from collections import defaultdict

import numpy as np

from .config import (
    CURVE_BREAK_MIN_RATIO,
    RRF_K,
)


def _minmax(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def _topk_positions(scores, topk):
    """The positions of the `topk` largest scores, descending.

    A FULL argsort (not an argpartition) on purpose: the old implementation
    used `np.argsort(-sims)[:topk]` and a partial sort breaks the ties
    differently -> the two rankings could differ in the tail.
    """
    n = scores.shape[0]
    if n == 0:
        return []
    return np.argsort(-scores)[:min(int(topk), n)]


def rrf_fusion(rankings, weights, k=RRF_K):
    fused = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, (idx, _score) in enumerate(ranking, start=1):
            fused[idx] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# --------------------------------------------------------------------------- #
# the curve estimators (shared by the document and the chunk stage)
# --------------------------------------------------------------------------- #

def _curve_break(rel):
    """The end of the relevant cluster, found with the BREAK RATIO.

    gap(i)   = rel[i] - rel[i+1]
    ratio(i) = gap(i) / gap(i+1)

    A smooth decay has a ratio ~ 1 everywhere; a cluster boundary has a big
    gap followed by small gaps -> a large ratio.  The estimator is scale free.

    Returns (index, ratio) or (0, ratio) when no break is found.
    """
    n = len(rel)
    if n < 5:
        return 0, 0.0
    gaps = rel[:-1] - rel[1:]
    if gaps.size < 2:
        return 0, 0.0
    # the denominator is guarded: a gap of 0 would give an infinite ratio
    floor = max(float(np.median(gaps)) * 0.1, 1e-9)
    ratios = gaps[:-1] / np.maximum(gaps[1:], floor)
    i = int(np.argmax(ratios))
    best = float(ratios[i])
    if best < CURVE_BREAK_MIN_RATIO:
        return 0, best
    return i + 1, best


def estimate_count(scores,
                   rel_cut=0.55,
                   break_w=0.5,
                   shrink=0.30,
                   floor=0,
                   lo=30,
                   hi=200,
                   target=120):
    """Estimate how many items of a score curve are relevant.

    `scores`   the scores, descending, all >= 0.
    `floor`    a hard lower bound (e.g. the cross-lingual consensus).
    `target`   the value the estimate is shrunk towards.
    `shrink`   how much of the raw estimate is kept (0 = always the target,
                1 = always the raw estimate).

    Returns (count, diagnostics).
    """
    n = len(scores)
    if n == 0:
        return 0, {'why': 'empty'}
    if n <= lo:
        return n, {'why': 'short'}

    hi = int(min(hi, n))
    s = np.asarray(scores[:hi], dtype=float)
    top = max(float(s[0]), 1e-12)
    rel = s / top

    # --- 1) threshold ------------------------------------------------------ #
    n_thresh = int(np.searchsorted(-rel, -float(rel_cut))) or 1
    n_thresh = max(n_thresh, 1)

    # --- 2) break ---------------------------------------------------------- #
    n_break, break_ratio = _curve_break(rel)

    # --- 3) combine -------------------------------------------------------- #
    if n_break:
        # the break is only trusted when it is roughly consistent with the
        # threshold (a factor of 3), otherwise the threshold wins
        if n_thresh <= max(2 * n_break, 2) or n_break <= 2 * n_thresh:
            n_raw = break_w * n_break + (1.0 - break_w) * n_thresh
        else:
            n_raw = n_thresh
    else:
        n_raw = n_thresh

    # --- 4) shrink towards the measured-good value ------------------------ #
    n_est = (1.0 - shrink) * float(target) + shrink * n_raw

    # the consensus of the languages is a hard lower bound, but it is capped
    # so that it cannot push the count above the pool
    n_est = max(n_est, min(float(floor), hi * 0.5))

    count = int(round(n_est))
    count = max(int(lo), min(count, hi))
    return count, {'why': 'break+thresh' if n_break else 'thresh',
                   'n_thresh': n_thresh, 'n_break': n_break,
                   'break_ratio': round(break_ratio, 2), 'floor': int(floor)}
