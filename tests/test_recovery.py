import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
import numpy as np
from memora.model import DEFAULT_PARAMS
from memora.fit import fit
from simulate import simulate_logs
from recovery import _replay_predictions, brier, reliability, TRUE


def test_simulator_spans_probability_range():
    """Jittered reviews must produce recall probabilities across (0,1), else nothing is
    identifiable."""
    logs = simulate_logs(TRUE, n_cards=200, n_reviews=8, seed=0)
    p, y = _replay_predictions(TRUE, logs)
    assert p.min() < 0.3 and p.max() > 0.9
    # outcomes are consistent with probabilities (higher p -> more recall)
    assert y[p > 0.8].mean() > y[p < 0.4].mean()


def test_behaviour_recovery_is_tight():
    """Fitting TRUE-generated logs recovers the true forgetting behaviour."""
    logs = simulate_logs(TRUE, n_cards=400, n_reviews=8, seed=1)
    theta, _ = fit(logs, x0=list(DEFAULT_PARAMS))
    test = simulate_logs(TRUE, n_cards=150, n_reviews=8, seed=1001)
    p_fit, _ = _replay_predictions(theta, test)
    p_true, _ = _replay_predictions(TRUE, test)
    assert np.mean(np.abs(p_fit - p_true)) < 0.03


def test_post_lapse_param_recovered():
    """The strongly-identified post-lapse-retention parameter (idx 7) is recovered."""
    logs = simulate_logs(TRUE, n_cards=500, n_reviews=8, seed=2)
    theta, _ = fit(logs, x0=list(DEFAULT_PARAMS))
    assert abs(theta[7] - TRUE[7]) < 0.08


def test_fitted_beats_naive_baselines_on_brier():
    train = simulate_logs(TRUE, n_cards=400, n_reviews=8, seed=3)
    test = simulate_logs(TRUE, n_cards=200, n_reviews=8, seed=303)
    theta, _ = fit(train, x0=list(DEFAULT_PARAMS))
    p_fit, y = _replay_predictions(theta, test)
    base_rate = _replay_predictions(TRUE, train)[1].mean()
    b_fit = brier(p_fit, y)
    b_const = brier(np.full_like(y, base_rate), y)
    assert b_fit < b_const * 0.6          # state-aware model is far better calibrated


def test_fitting_does_not_worsen_calibration_vs_default():
    train = simulate_logs(TRUE, n_cards=400, n_reviews=8, seed=4)
    test = simulate_logs(TRUE, n_cards=200, n_reviews=8, seed=404)
    theta, _ = fit(train, x0=list(DEFAULT_PARAMS))
    p_fit, y = _replay_predictions(theta, test)
    p_def, _ = _replay_predictions(DEFAULT_PARAMS, test)
    assert brier(p_fit, y) <= brier(p_def, y) + 1e-3


def test_reliability_near_diagonal():
    train = simulate_logs(TRUE, n_cards=500, n_reviews=8, seed=5)
    test = simulate_logs(TRUE, n_cards=300, n_reviews=8, seed=505)
    theta, _ = fit(train, x0=list(DEFAULT_PARAMS))
    p, y = _replay_predictions(theta, test)
    xs, ys, ns = reliability(p, y, bins=10)
    assert np.mean(np.abs(xs - ys)) < 0.06      # calibrated: on the diagonal
