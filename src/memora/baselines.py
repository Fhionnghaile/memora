"""Baseline schedulers for the benchmark: SM-2 and a compact FSRS-style model."""
from __future__ import annotations
import math

class SM2:
    """Classic SuperMemo-2. We expose a recall predictor by treating the scheduled
    interval as the point where retention = `target`, giving an implied stability."""
    def __init__(self, target=0.9): self.target = target
    def implied_recall(self, interval_days, elapsed_days):
        if interval_days <= 0: return 0.5
        S = -interval_days / math.log(self.target)       # exp model implied by the interval
        return math.exp(-elapsed_days / S)
    @staticmethod
    def update(ef, reps, interval, quality):
        if quality < 3: return (max(1.3, ef), 0, 1.0)
        ef = max(1.3, ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        if reps == 0: interval = 1.0
        elif reps == 1: interval = 6.0
        else: interval = interval * ef
        return (ef, reps + 1, interval)

class FSRSLite:
    """Power-law forgetting curve R=(1+F*t/S)^D, the FSRS-4.5 shape, fixed params."""
    F, D = 19.0 / 81.0, -0.5
    def recall(self, S, t): return (1.0 + self.F * t / S) ** self.D
