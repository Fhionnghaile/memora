"""The DSR scheduler core. Pure, typed, dependency-free (numpy only for fit/bench)."""
from __future__ import annotations
from dataclasses import dataclass
import math

# Parameter vector (all fittable). Indices documented for fit.py.
#  0..3  initial stability (days) for grade 1,2,3,4
#  4     initial difficulty D0 (grade 3)
#  5     difficulty slope per grade step on init
#  6     stability-growth log-coefficient (w_a)
#  7     post-lapse stability retention (0..1)
#  8     difficulty increase on a lapse
#  9     difficulty decrease per grade step on success
DEFAULT_PARAMS = [0.4, 1.2, 3.0, 7.0, 5.0, 0.9, -0.2, 0.30, 1.4, 0.6]
_GRADE_MULT = {2: 0.6, 3: 1.0, 4: 1.4}  # hard / good / easy weighting on growth


def _clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class Card:
    difficulty: float   # 1 (easy) .. 10 (hard)
    stability: float    # days; time for retrievability to fall to 1/e
    last_review: float  # day index of the last review (float days)
    reps: int = 0

    def retrievability(self, now: float) -> float:
        """Exponential forgetting: R(t) = exp(-t/S), in (0, 1]."""
        t = max(0.0, now - self.last_review)
        return math.exp(-t / self.stability)


class Scheduler:
    def __init__(self, params=None):
        self.p = list(DEFAULT_PARAMS if params is None else params)

    # --- lifecycle -----------------------------------------------------------
    def init_card(self, grade: int, now: float = 0.0) -> Card:
        if grade < 1 or grade > 4: raise ValueError("grade must be 1..4")
        s0 = max(0.1, self.p[grade - 1])
        d0 = _clamp(self.p[4] - self.p[5] * (grade - 3), 1.0, 10.0)
        return Card(difficulty=d0, stability=s0, last_review=now, reps=1)

    def review(self, card: Card, grade: int, now: float) -> Card:
        if grade < 1 or grade > 4: raise ValueError("grade must be 1..4")
        R = card.retrievability(now)
        D, S = card.difficulty, card.stability
        if grade == 1:  # lapse
            S_new = max(0.1, self.p[7] * S * (0.5 + 0.5 * (1 - R)))
            D_new = _clamp(D + self.p[8], 1.0, 10.0)
        else:           # success
            growth = 1.0 + math.exp(self.p[6]) * (11.0 - D) * (1.0 - R) * _GRADE_MULT[grade]
            S_new = S * growth
            D_new = _clamp(D - self.p[9] * (grade - 3), 1.0, 10.0)
        return Card(difficulty=D_new, stability=S_new, last_review=now, reps=card.reps + 1)

    # --- scheduling ----------------------------------------------------------
    def next_interval(self, card: Card, request_retention: float = 0.9) -> float:
        """Days until retrievability decays to `request_retention` (>=1 day)."""
        if not 0 < request_retention < 1: raise ValueError("retention in (0,1)")
        return max(1.0, -card.stability * math.log(request_retention))

    def predict_recall(self, card: Card, now: float) -> float:
        return card.retrievability(now)
