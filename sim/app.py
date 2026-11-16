"""Memora live simulator (Streamlit).  Run: streamlit run sim/app.py"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import streamlit as st
from memora import Scheduler

st.set_page_config(page_title="Memora simulator", layout="wide")
st.title("Memora - DSR spaced-repetition simulator")

sched = Scheduler()
retention = st.sidebar.slider("Target retention", 0.70, 0.98, 0.90, 0.01)
grade = st.sidebar.select_slider("Review grade", options=[1, 2, 3, 4], value=3,
                                 format_func=lambda g: {1:"Again",2:"Hard",3:"Good",4:"Easy"}[g])
n_reviews = st.sidebar.slider("Reviews to project", 1, 12, 8)

card = sched.init_card(3, now=0.0)
rows, now = [], 0.0
for i in range(n_reviews):
    interval = sched.next_interval(card, retention)
    now += interval
    rows.append({"review": i + 1, "interval_days": round(interval, 2),
                 "due_day": round(now, 1), "stability": round(card.stability, 2),
                 "difficulty": round(card.difficulty, 2)})
    card = sched.review(card, grade, now)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Forgetting curve (current card)")
    t = np.linspace(0, max(1.0, sched.next_interval(card, retention) * 1.5), 100)
    S = card.stability
    st.line_chart({"R(t)": np.exp(-t / S)})
with c2:
    st.subheader("Projected due-queue")
    st.dataframe(rows, use_container_width=True)
st.caption("Intervals grow as stability grows; a lapse (Again) shrinks stability and the next interval.")
