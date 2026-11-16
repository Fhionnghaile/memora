import math
from memora import Scheduler, Card, fit, bce, DEFAULT_PARAMS

S = Scheduler()

def test_retrievability_bounds_and_monotonic():
    c = S.init_card(3, now=0)
    rs = [c.retrievability(t) for t in range(0, 30)]
    assert rs[0] == 1.0
    assert all(0 < r <= 1 for r in rs)
    assert all(rs[i] >= rs[i+1] for i in range(len(rs)-1))

def test_success_grows_stability_lapse_shrinks():
    c = S.init_card(3, now=0)
    grown = S.review(c, 3, now=S.next_interval(c))
    assert grown.stability > c.stability
    lapsed = S.review(grown, 1, now=grown.last_review + S.next_interval(grown))
    assert lapsed.stability < grown.stability
    assert lapsed.difficulty > grown.difficulty

def test_interval_increases_with_stability_and_formula():
    c = S.init_card(3, now=0)
    easy = Card(difficulty=c.difficulty, stability=c.stability*4, last_review=0)
    assert S.next_interval(easy) > S.next_interval(c)
    assert math.isclose(S.next_interval(Card(5, 10, 0)), -10*math.log(0.9), rel_tol=1e-9)

def test_grade_ordering():
    c = S.init_card(3, 0); t = S.next_interval(c)
    hard = S.review(c, 2, t).stability
    good = S.review(c, 3, t).stability
    easy = S.review(c, 4, t).stability
    assert hard < good < easy

def _synth_logs(n_cards=250, seed=0):
    import random; rng = random.Random(seed); logs = []
    for cid in range(n_cards):
        S_true = rng.uniform(0.5, 3.0); last = 0.0; logs.append((cid, 3, 0.0, True))
        for _ in range(12):
            gap = max(1.0, S_true * rng.uniform(0.5, 1.5))
            R = math.exp(-gap / S_true); recalled = rng.random() < R
            logs.append((cid, 3 if recalled else 1, gap, recalled))
            S_true = S_true * 1.6 if recalled else max(0.4, S_true * 0.5)
    return logs

def test_fit_improves_or_matches_default():
    logs = _synth_logs()
    params, loss = fit(logs)
    # default-parameter loss on the same logs
    from memora.fit import _trajectory_loss
    assert loss <= _trajectory_loss(DEFAULT_PARAMS, logs) + 1e-9
    assert len(params) == len(DEFAULT_PARAMS)
