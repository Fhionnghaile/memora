"""Parameter-recovery + calibration study for the Memora DSR model.

This is the *honest* way to validate a forgetting model without human subjects:

  1. PARAMETER RECOVERY — generate logs from a known ground truth, fit from a different
     start, and show the fitted model reproduces the true forgetting BEHAVIOUR (and the
     identifiable parameters), across many seeds (bias/variance).

  2. CALIBRATION — on a held-out test set, do predicted recall probabilities match
     observed recall frequencies? Reported as a reliability diagram and a Brier score,
     against baselines. A model can be accurate-on-average yet badly calibrated; we check.

Everything here runs on SYNTHETIC logs. The same harness accepts real review logs
unchanged — swap `simulate_logs` for your data and the recovery step for the calibration
step alone.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from memora.model import Scheduler, DEFAULT_PARAMS
from memora.fit import fit
from simulate import simulate_logs

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# a ground truth DELIBERATELY far from DEFAULT_PARAMS (esp. the growth and post-lapse
# terms that drive predictions), so a default model mispredicts and fitting must earn its keep
TRUE = [0.6, 1.0, 1.8, 9.0, 4.5, 0.8, 0.2, 0.55, 1.8, 0.4]

# Each parameter with (human name, what data informs it). The last column is the honest
# story: a parameter is only as identifiable as the events that exercise it.
IDENTIFIABLE = {
    7: ("post-lapse retention", "every lapse", "strong"),
    6: ("log stability-growth", "every success", "strong"),
    4: ("D0 (init difficulty)", "one event/card (init)", "moderate"),
    8: ("lapse difficulty +", "every lapse", "moderate"),
    9: ("success difficulty -", "grade-4 successes only (~25%)", "weak"),
}


def _replay_predictions(params, logs):
    """Replay logs under `params`; return (pred_recall, outcome) for every non-init review."""
    sched = Scheduler(params)
    cards, last_t, preds, ys = {}, {}, [], []
    for cid, grade, elapsed, recalled in logs:
        if cid not in cards:
            cards[cid] = sched.init_card(grade, 0.0); last_t[cid] = 0.0
            continue
        now = last_t[cid] + elapsed
        preds.append(sched.predict_recall(cards[cid], now)); ys.append(float(recalled))
        cards[cid] = sched.review(cards[cid], grade, now); last_t[cid] = now
    return np.array(preds), np.array(ys)


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def reliability(p, y, bins=10):
    p = np.asarray(p); y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() >= 5:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def recovery_study(seeds=12):
    """Fit over many seeds; return per-parameter fitted distribution and behaviour MAE."""
    fits = []
    behav_mae = []
    # dense grid of true predicted-recall vs fitted predicted-recall, on held-out states
    for s in seeds_iter(seeds):
        logs = simulate_logs(TRUE, n_cards=400, n_reviews=8, seed=s)
        theta, _ = fit(logs, x0=list(DEFAULT_PARAMS))
        fits.append(theta)
        # behaviour recovery: predictions of fitted vs TRUE on an independent test set
        test = simulate_logs(TRUE, n_cards=150, n_reviews=8, seed=1000 + s)
        p_fit, _ = _replay_predictions(theta, test)
        p_true, _ = _replay_predictions(TRUE, test)
        behav_mae.append(np.mean(np.abs(p_fit - p_true)))
    return np.array(fits), np.array(behav_mae)


def seeds_iter(n):
    return list(range(n))


def main():
    # ---- 1. recovery ----
    fits, behav_mae = recovery_study(seeds=12)
    mean, std = fits.mean(0), fits.std(0)

    # ---- 2. calibration on a fresh train/test split ----
    train = simulate_logs(TRUE, n_cards=500, n_reviews=8, seed=7)
    test = simulate_logs(TRUE, n_cards=300, n_reviews=8, seed=99)
    theta, _ = fit(train, x0=list(DEFAULT_PARAMS))

    p_fit, y = _replay_predictions(theta, test)
    p_def, _ = _replay_predictions(DEFAULT_PARAMS, test)
    base_rate = float(np.mean(_replay_predictions(TRUE, train)[1]))   # train recall rate
    p_const = np.full_like(y, base_rate)
    # global single-stability exponential baseline fitted on train (ignores card state)
    p_glob = _global_exp_baseline(train, test)

    models = {"Memora (fitted)": p_fit, "Memora (default, unfitted)": p_def,
              "Global 1-param exp": p_glob, "Constant base-rate": p_const}
    briers = {k: brier(v, y) for k, v in models.items()}

    _plots(fits, mean, std, p_fit, y, briers)
    _summary(mean, std, behav_mae, briers)
    print(open(os.path.join(OUT, "recovery_summary.md")).read())


def _global_exp_baseline(train, test):
    """Fit one stability S (R=exp(-t/S)) to train elapsed/outcomes; predict on test."""
    # gather (elapsed, recalled) pairs from the per-card trajectory (elapsed since last)
    def pairs(logs):
        last = {}; out = []
        for cid, g, e, r in logs:
            if cid not in last:
                last[cid] = 0.0; continue
            out.append((e, r));
        return out
    tr = pairs(train)
    es = np.array([e for e, _ in tr]); rs = np.array([r for _, r in tr])
    from scipy.optimize import minimize_scalar
    def nll(S):
        p = np.clip(np.exp(-es / max(S, 1e-3)), 1e-6, 1 - 1e-6)
        return -np.mean(rs * np.log(p) + (1 - rs) * np.log(1 - p))
    S = minimize_scalar(nll, bounds=(0.5, 200), method="bounded").x
    te = pairs(test)
    return np.clip(np.exp(-np.array([e for e, _ in te]) / S), 1e-6, 1 - 1e-6)


def _plots(fits, mean, std, p_fit, y, briers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)

    # reliability diagram
    xs, ys, ns = reliability(p_fit, y, bins=10)
    ax1.plot([0, 1], [0, 1], "--", color="#999999", lw=1.4, label="perfect calibration")
    ax1.plot(xs, ys, "-o", color="#1f6feb", lw=2.2, label=f"Memora (Brier {briers['Memora (fitted)']:.3f})")
    ax1.set_xlabel("Predicted recall probability"); ax1.set_ylabel("Observed recall frequency")
    ax1.set_title("Calibration on held-out reviews"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.grid(True, color="#eee"); ax1.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"): ax1.spines[sp].set_visible(False)

    # Brier comparison
    names = list(briers.keys()); vals = [briers[n] for n in names]
    colors = ["#1f6feb", "#8957e5", "#d08c00", "#999999"]
    ax2.barh(range(len(names)), vals, color=colors[:len(names)])
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis(); ax2.set_xlabel("Brier score (lower is better)")
    ax2.set_title("Memora vs baselines")
    for i, v in enumerate(vals): ax2.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "recovery.png"))


def _summary(mean, std, behav_mae, briers):
    with open(os.path.join(OUT, "recovery_summary.md"), "w") as f:
        f.write("# Memora — parameter recovery & calibration (synthetic)\n\n")
        f.write(f"**Behaviour recovery:** fitted model's predicted recall matches the true "
                f"model within **MAE {behav_mae.mean():.3f}** (±{behav_mae.std():.3f}) over "
                f"12 seeds, on held-out reviews.\n\n")
        f.write("**Parameter recovery** (ground truth vs fitted mean ± sd, 12 seeds). A "
                "parameter is only as recoverable as the events that exercise it — shown "
                "in the last column:\n\n")
        f.write("| Parameter | True | Fitted | Informed by | Identifiability |\n|---|--:|--:|---|---|\n")
        for idx, (name, informed, strength) in IDENTIFIABLE.items():
            f.write(f"| {name} | {TRUE[idx]:.2f} | {mean[idx]:.2f} ± {std[idx]:.2f} | {informed} | {strength} |\n")
        f.write("\n**Calibration (held-out Brier, lower is better):**\n\n")
        f.write("| Model | Brier |\n|---|--:|\n")
        for k, v in sorted(briers.items(), key=lambda kv: kv[1]):
            f.write(f"| {k} | {v:.3f} |\n")
        f.write("\n*Synthetic logs; the same harness accepts real review logs unchanged. "
                "Parameters not exercised by this data regime (e.g. per-grade initial "
                "stabilities for unused grades) are not claimed as identified — only the "
                "behaviour and the parameters the data informs.*\n")


if __name__ == "__main__":
    # make TRUE visible to _summary
    globals()["TRUE"] = TRUE
    main()
