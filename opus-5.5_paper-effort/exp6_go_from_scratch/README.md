# Experiment 6: write a 1.5M-weight 9x9 Go network from scratch (Claude Opus 5.5, effort xhigh)

Original: `exp6_go_from_scratch/`. Harness: `harness/exp6/`.

- `runs/budget_<B>/`: the four conversations (pass budgets 0, 10, 25, 50; up to 120 Python runs). `requests/` holds
  the pass requests. `work/` holds what Opus wrote into its `/work` folder, filtered as in the original release:
  design scripts and coefficient files (`*.py`, `*.json`, `*.pkl`), and per pass `pass_<k>/summary.json` +
  `README.txt` (games pass) or the complete outputs (positions pass). Left out: the per-game records of games
  passes (`games.json`, `boards.npy`, `valid.npy`, 190 MB) and the whole-network copy in `work/net/` (identical to
  `networks/`); see `EXCLUDED.md`. The `*.pkl` files are Python pickles written by the model: load them only if you
  trust them.
- `networks/budget_<B>/`: the final networks (14 tensors, `arch.json`, `sha256.txt`).
- `eval/budget_<B>/`: the hidden evaluation (`eval_submission.json`), head-to-heads against the released same-size
  PPO networks (`h2h_*.json`), 500 games against the practice "strong" opponent (`analysis_strong.json`), and
  `inputs.json` (every file the scorer loaded, with its SHA-256). `eval/scores.json`, `eval/summary.md`.
- The opponents we trained (the original's were not released) are shared by all model runs: `../../exp6_opponents/`
  (see its README).

| budget | vs our PPO s0 / s2 (mean) | Astra's released network vs the same opponents | h2h vs released same-size PPO s0 / s1 / s2 |
|---|---|---|---|
| 0 | 65.6 % | 37.5 % | 60.1 / 56.4 / 69.7 (Astra 43.0 / 26.4 / 35.6) |
| 10 | 67.9 % | 62.8 % | 69.9 / 60.5 / 72.8 (Astra 43.1 / 45.8 / 57.3) |
| 25 | 63.4 % | 54.7 % | 55.7 / 46.3 / 58.7 (Astra 46.9 / 43.8 / 50.7) |
| 50 | 58.1 % | 78.4 % | 54.7 / 47.5 / 59.5 (Astra 72.8 / 67.0 / 80.3) |

Absolute win rates against our opponents are not the paper's (our PPO opponents are about 6 points stronger on
average); compare with Astra's networks re-scored against them (`../../validation/astra-rescored/exp6/`).
