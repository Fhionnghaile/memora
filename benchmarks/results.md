# Memora benchmark (generated)

Synthetic exponential-forgetting logs; 400 train / 150 test cards; DSR fitted on train.

| model | BCE (lower=better) | AUC (higher=better) |
|---|---|---|
| DSR (fitted) | 0.4698 | 0.703 |
| DSR (default) | 0.7858 | 0.757 |
| FSRS-lite | 1.5107 | 0.763 |
| SM-2 | 1.8003 | 0.379 |

DSR train BCE after fit: 0.4511. Reproduce: `PYTHONPATH=src python benchmarks/run.py`.
