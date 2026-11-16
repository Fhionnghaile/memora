"""Fit DSR parameters to recall logs by minimising binary cross-entropy."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .model import Scheduler, Card, DEFAULT_PARAMS

# A review log entry: (card_id, grade, elapsed_days, recalled)  in chronological order.
_EPS = 1e-6

def bce(probs, outcomes) -> float:
    p = np.clip(np.asarray(probs, float), _EPS, 1 - _EPS)
    y = np.asarray(outcomes, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def _trajectory_loss(params, logs) -> float:
    sched = Scheduler(params)
    cards: dict = {}
    last_t: dict = {}
    probs, ys = [], []
    for cid, grade, elapsed, recalled in logs:
        if cid not in cards:
            cards[cid] = sched.init_card(grade, now=0.0); last_t[cid] = 0.0
            continue
        now = last_t[cid] + elapsed
        probs.append(sched.predict_recall(cards[cid], now)); ys.append(float(recalled))
        cards[cid] = sched.review(cards[cid], grade, now); last_t[cid] = now
    if not probs: return 1e3
    return bce(probs, ys)

def fit(logs, x0=None, bounds=None):
    """Return fitted params minimising recall BCE over the logs."""
    x0 = list(DEFAULT_PARAMS if x0 is None else x0)
    if bounds is None:
        bounds = [(0.1, 10)]*4 + [(1, 10), (0.0, 2.0), (-3, 3), (0.05, 0.95), (0.0, 3.0), (0.0, 2.0)]
    res = minimize(_trajectory_loss, x0, args=(logs,), method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200})
    return list(res.x), float(res.fun)
