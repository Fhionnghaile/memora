"""Benchmark DSR (fitted) vs default DSR, SM-2 and FSRS-lite on synthetic recall logs.
Ground truth: exponential forgetting; stability grows on success, shrinks on lapse.
Run: PYTHONPATH=src python benchmarks/run.py"""
from __future__ import annotations
import os, sys, math, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from memora import Scheduler, fit
from memora.fit import bce
from memora.baselines import SM2, FSRSLite

def gen(n_cards, seed):
    rng = random.Random(seed); logs = []
    for cid in range(n_cards):
        S_true = rng.uniform(0.5, 3.0); logs.append((cid, 3, 0.0, True))
        for _ in range(14):
            gap = max(1.0, S_true * rng.uniform(0.4, 1.6))
            recalled = rng.random() < math.exp(-gap / S_true)
            logs.append((cid, 3 if recalled else 1, gap, recalled))
            S_true = S_true * 1.6 if recalled else max(0.4, S_true * 0.5)
    return logs

def auc(probs, ys):
    p = np.asarray(probs); y = np.asarray(ys)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); order = np.argsort(allv)
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

def eval_dsr(params, logs):
    s=Scheduler(params); cards={}; last={}; P=[]; Y=[]
    for cid,g,el,rec in logs:
        if cid not in cards: cards[cid]=s.init_card(g,0.0); last[cid]=0.0; continue
        now=last[cid]+el; P.append(s.predict_recall(cards[cid],now)); Y.append(float(rec))
        cards[cid]=s.review(cards[cid],g,now); last[cid]=now
    return P,Y

def eval_sm2(logs):
    sm=SM2(); st={}; P=[]; Y=[]
    for cid,g,el,rec in logs:
        if cid not in st: st[cid]=[2.5,0,1.0]; continue
        ef,reps,interval=st[cid]; P.append(sm.implied_recall(interval,el)); Y.append(float(rec))
        st[cid]=list(SM2.update(ef,reps,interval,4 if rec else 2))
    return P,Y

def eval_fsrs(logs):
    f=FSRSLite(); S={}; P=[]; Y=[]
    for cid,g,el,rec in logs:
        if cid not in S: S[cid]=2.0; continue
        P.append(f.recall(S[cid],el)); Y.append(float(rec))
        S[cid]=S[cid]*1.7 if rec else max(0.4,S[cid]*0.5)
    return P,Y

train, test = gen(400,1), gen(150,99)
params, train_loss = fit(train)
models={"DSR (fitted)":eval_dsr(params,test),"DSR (default)":eval_dsr(None,test),
        "SM-2":eval_sm2(test),"FSRS-lite":eval_fsrs(test)}
rows=[(n,bce(P,Y),auc(P,Y)) for n,(P,Y) in models.items()]
rows=sorted(rows,key=lambda r:r[1])
print(f"{'model':16}{'BCE':>9}{'AUC':>8}")
for n,b,a in rows: print(f"{n:16}{b:9.4f}{a:8.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C={"DSR (fitted)":"#1b7837","DSR (default)":"#a6dba0","SM-2":"#2166ac","FSRS-lite":"#b2182b"}
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.2))
ts=np.linspace(0,20,100)
for S_,lab,c in [(2,"S=2d","#2166ac"),(6,"S=6d","#1b7837"),(14,"S=14d","#b35806")]:
    ax1.plot(ts,np.exp(-ts/S_),label=lab,color=c,lw=2)
ax1.axhline(0.9,ls="--",c="#888",lw=1); ax1.text(0.3,0.915,"target 90%",fontsize=8,color="#555")
ax1.set(title="DSR forgetting curves",xlabel="days since review",ylabel="retrievability R(t)"); ax1.legend(frameon=False); ax1.set_ylim(0,1.02)
names=[r[0] for r in rows]; bces=[r[1] for r in rows]
ax2.barh(names,bces,color=[C[n] for n in names]); ax2.invert_yaxis()
for i,b in enumerate(bces): ax2.text(b+0.003,i,f"{b:.3f}",va="center",fontsize=9)
ax2.set(title="Held-out recall-prediction loss (lower=better)",xlabel="binary cross-entropy"); ax2.set_xlim(0,max(bces)*1.18)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"..","docs","benchmark.png"),dpi=130)
print("saved docs/benchmark.png")
with open(os.path.join(os.path.dirname(__file__),"results.md"),"w") as f:
    f.write("# Memora benchmark (generated)\n\nSynthetic exponential-forgetting logs; 400 train / 150 test cards; DSR fitted on train.\n\n")
    f.write("| model | BCE (lower=better) | AUC (higher=better) |\n|---|---|---|\n")
    for n,b,a in rows: f.write(f"| {n} | {b:.4f} | {a:.3f} |\n")
    f.write(f"\nDSR train BCE after fit: {train_loss:.4f}. Reproduce: `PYTHONPATH=src python benchmarks/run.py`.\n")
print("wrote benchmarks/results.md")
