# Experiment 6 opponents (trained by us; the original's were not released)

| Path | Contents |
|---|---|
| `<name>/` | the six final opponents, 243-32-32-82 tanh MLPs (`L1.W.npy` ... `out.b.npy`, `arch.json`): `ppo_s1` = practice "strong", `reinforce_s1` = practice "weak", `ppo_s0`, `ppo_s2`, `reinforce_s0`, `reinforce_s2` = hidden scoring opponents |
| `manifest.sha256` | SHA-256 of every opponent file (checked before every load). The `source` path in each `arch.json` was replaced by a placeholder for this package, so the six `arch.json` lines were updated; the weight files and their hashes are unchanged |
| `strength.json`, `strength.md` | calibration of the final opponents against the original ones (2,000 games per released reference network) |
| `recipe_search/<variant>/<name>/` | the recipe search v2-v8 and the final v4 runs: `config.json`, `train_log.jsonl`, `eval_log.jsonl`, `calib_<iteration>.json` (checkpoints not included) |
| `training_logs/` | the training, screening, benchmark and GPU-time logs |

Not included: the training checkpoints (`iter*.npz`, `state.pkl`, 27 MB) and the archived first attempt
(`opponents_v1`, recipe v1, stopped at iteration 7,000; its numbers are in the recipe-search table below). Code:
`harness/exp6/train_opponents.py`, `calibration.py`, `scripts/`.

The rest of this file is the recipe-search and final-opponent part of `harness/exp6/README.md`.

## Opponents (not released; trained by us)

Six small policies, `~/wsp/exp6_data/opponents/<name>/` (L1.W ... out.b + arch.json `{"depth": 2, "width": 32,
"act": "tanh"}`):

| ours | role in the original | used by |
|---|---|---|
| `ppo_s1` | practice "strong" ("a third PPO network, seed 1") | games pass, analysis_strong |
| `reinforce_s1` | practice "weak" ("a REINFORCE network") | games pass |
| `ppo_s0`, `ppo_s2` | hidden scoring opponents of the post | score.py |
| `reinforce_s0`, `reinforce_s2` | hidden opponents in `main_score`, not in the post | score.py |

What the original tells us, and where:
- 243 -> 32 -> 32 -> 82, PPO, 20,000 iterations of 1,024 games (= 20.5 M games) (`exp6_go_from_scratch/README.md`,
  the blog post).
- Self-play with an opponent pool: the prompt says the hidden opponents were "trained with reinforcement learning on
  this game (self-play)"; `eval/ppo_same_size/ppo_big/*/ref.json` says a resume "refilled" the "opponent pool" and
  reset the Adam moments.
- Checkpoints `iter<k>.npz` every 1,000 iterations with keys `actor/dense1/kernel ... actor/logits/kernel`, a
  separate critic (`eval_ppo_big.py`); the same layout as the authors' Exp 1 PPO networks, whose
  `networks/meta/*.json` give their PPO settings: CleanRL/PureJaxRL style, tanh, orthogonal init (hidden gain sqrt 2,
  actor output 0.01, critic output 1.0), lr annealed, 4 epochs x 8 minibatches, gamma 0.99, GAE 0.95, clip 0.2,
  entropy 0.01, value 0.5, grad norm 0.5.
- Their Go script (`gomatch train.py`) needed `--act relu` for the same-size network, so its default activation is
  not ReLU; we take tanh, as in all their other PPO networks. Its untuned learning rate was 3e-4
  (`exp6_go_from_scratch/README.md`: "the original, untuned settings (lr 3e-4)").
- Strength: practice "weak" and "strong" beat a random player 95 % and 98 %; the hidden ones 79-98 % (prompt).
- Nothing about REINFORCE beyond its name.

### Recipe search (calibration against the original opponents, released data only)

`calibration.py` derives targets from released files only: the original opponents' win rates against seven released
networks (Astra's budget 0/10/25/50 networks: `eval/budget_*/eval_submission.json`, `analysis_strong.json`; the
same-size PPO s0/s1/s2: `eval_lr_scan.json`) are 1 - those networks' scores; plus the prompt's vs-random rates
("strong" 98 %, "weak" 95 %). Nothing the studied model produces is used. Our opponent plays the same seven networks
(1,000 games each during the search, 2,000 for the final check; Black in half).

Original targets (%; per opponent, and the family means used in the search):

| original opponent | Astra b0 | Astra b10 | Astra b25 | Astra b50 | PPO-512 s0 | PPO-512 s1 | PPO-512 s2 | random |
|---|---|---|---|---|---|---|---|---|
| ppo_s0 | 45.9 | 27.1 | 24.0 | 16.0 | 48.0 | 26.7 | 44.1 | - |
| ppo_s1 (strong) | 53.0 | 24.0 | 25.4 | 16.2 | 25.8 | 37.2 | 49.4 | 98.0 |
| ppo_s2 | 45.1 | 40.0 | 37.7 | 19.2 | 52.1 | 23.9 | 44.6 | - |
| reinforce_s0 | 1.2 | 1.0 | 2.0 | 0.2 | 14.5 | 7.0 | 9.9 | - |
| reinforce_s1 (weak) | - | - | - | - | - | - | - | 95.0 |
| reinforce_s2 | 0.3 | 0.1 | 1.9 | 0.1 | 2.0 | 4.5 | 5.5 | - |
| PPO family mean | 48.0 | 30.4 | 29.1 | 17.1 | 42.0 | 29.3 | 46.0 | 98.0 |
| REINFORCE family mean | 0.8 | 0.5 | 1.9 | 0.1 | 8.2 | 5.8 | 7.7 | 95.0 |

The original seeds differ from each other by up to 22 points on the same reference (e.g. same-size s0: PPO s0 48.0,
"strong" 25.8, PPO s2 52.0), so no single recipe can put every seed within 5 points of its own seed's numbers; the
search therefore targets the family mean per reference (mean over the original PPO s0/s1/s2, resp. REINFORCE s0/s2),
and the final table also lists each seed against its own targets.

Every variant tried (PPO "strong" role, seed 1; full calibration at the iterations shown; "mean diff" = mean over
the 7 reference networks of ours minus the family target, in points). All runs use the fixed schedule (20,000
iterations x 1,024 games, lr annealed over 20,000); screening = the first N iterations of that schedule:

| variant | network | iteration | Astra b0 | Astra b10 | Astra b25 | Astra b50 | PPO-512 s0 | PPO-512 s1 | PPO-512 s2 | random | mean diff vs family (7 refs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4 | ppo_s0 | 20,000 | 65.6 | 37.4 | 48.3 | 20.9 | 37.7 | 34.7 | 40.8 | 99.0 | +6.2 |
| v1 | ppo_s1 | 5,000 | 6.0 | 4.3 | 8.9 | 0.7 | 15.5 | 5.9 | 12.5 | 95.4 | -26.9 |
| v1 | ppo_s1 | 7,000 | 12.6 | 5.4 | 7.2 | 2.7 | 13.2 | 5.2 | 9.2 | 87.4 | -26.6 |
| v2 | ppo_s1 | 5,000 | 32.5 | 22.4 | 23.4 | 14.2 | 23.3 | 13.3 | 16.9 | 98.4 | -13.7 |
| v2 | ppo_s1 | 10,000 | 56.8 | 46.3 | 44.5 | 12.4 | 49.2 | 18.6 | 61.7 | 99.9 | +6.8 |
| v2 | ppo_s1 | 15,000 | 65.2 | 33.4 | 41.3 | 17.0 | 40.7 | 36.8 | 21.2 | 99.5 | +2.0 |
| v2 | ppo_s1 | 20,000 | 60.3 | 36.5 | 41.0 | 15.3 | 45.2 | 41.7 | 30.4 | 99.0 | +4.1 |
| v3 | ppo_s1 | 5,000 | 31.1 | 33.6 | 28.8 | 12.4 | 16.2 | 8.7 | 11.7 | 94.2 | -14.2 |
| v4 | ppo_s1 | 5,000 | 46.7 | 28.4 | 32.2 | 17.8 | 19.2 | 14.4 | 24.4 | 99.2 | -8.4 |
| v4 | ppo_s1 | 10,000 | 60.5 | 48.9 | 48.8 | 14.7 | 60.6 | 25.8 | 75.1 | 99.7 | +13.2 |
| v4 | ppo_s1 | 15,000 | 66.1 | 46.8 | 47.9 | 16.3 | 46.8 | 22.9 | 31.3 | 99.4 | +5.2 |
| v4 | ppo_s1 | 20,000 | 63.6 | 38.2 | 33.2 | 15.0 | 43.3 | 22.5 | 49.8 | 99.2 | +3.4 |
| v5 | ppo_s1 | 5,000 | 25.7 | 42.0 | 35.8 | 6.0 | 5.7 | 9.9 | 24.1 | 98.5 | -13.2 |
| v6 | ppo_s1 | 5,000 | 67.4 | 46.1 | 40.6 | 19.6 | 36.8 | 44.6 | 75.6 | 98.3 | +12.7 |
| v7 | ppo_s1 | 5,000 | 51.6 | 31.8 | 36.5 | 8.1 | 34.0 | 36.4 | 48.2 | 99.2 | +0.7 |
| v8 | ppo_s1 | 5,000 | 0.6 | 1.4 | 7.1 | 0.1 | 7.9 | 6.6 | 9.1 | 71.9 | -29.9 |
| v8 | ppo_s1 | 10,000 | 1.3 | 0.7 | 2.8 | 0.1 | 10.0 | 9.6 | 8.4 | 70.3 | -29.8 |
| v4 | ppo_s2 | 20,000 | 60.8 | 33.3 | 43.6 | 25.3 | 19.2 | 52.1 | 50.8 | 99.5 | +6.2 |
| v4 | reinforce_s0 | 20,000 | 0.7 | 1.1 | 3.1 | 0.0 | 8.2 | 7.5 | 7.8 | 65.0 | +0.5 |
| v4 | reinforce_s1 | 5,000 | 1.0 | 0.9 | 1.7 | 0.3 | 12.2 | 8.9 | 13.2 | 92.0 | +1.9 |
| v4 | reinforce_s1 | 10,000 | 1.2 | 0.7 | 2.5 | 0.0 | 9.4 | 7.9 | 12.6 | 92.9 | +1.3 |
| v4 | reinforce_s1 | 20,000 | 1.8 | 1.2 | 2.9 | 0.1 | 9.8 | 8.6 | 10.2 | 90.8 | +1.4 |
| v4 | reinforce_s2 | 20,000 | 2.0 | 0.8 | 2.2 | 0.1 | 6.9 | 5.5 | 11.0 | 92.2 | +0.5 |

| variant | change | outcome |
|---|---|---|
| v1 | first guess: Exp 1/5 PPO defaults, learner's moves only, opponent uniform over current + 16 snapshots | plateau at -27 (stopped at 7,000) |
| v2 | train on both sides of self-play games, current policy as opponent in 50 % of games (else a recent snapshot) | -13.7 at 5k, +4.1 at 20k |
| v3 | v2 + entropy 0.001 | no better than v2 at 5k; worse vs random |
| v4 | v2 + gamma 1 (terminal reward, no discount) | -8.4 at 5k, +13.2 at 10k, +5.2 at 15k, **+3.4 at 20k** |
| v5 | v2 + ReLU | no better than v2 at 5k |
| v6 | v2 + lr 1e-3 | +12.7 already at 5k (too strong) |
| v7 | v2 with pure self-play (no pool) | +0.7 already at 5k (would overshoot) |
| v8 | v4 + lr 1e-4 | -30 at 5k and 10k (stopped) |

**Chosen recipe: v4** for both algorithms (the hidden opponents use exactly the practice opponents' recipe, other
seeds). It keeps every known setting of the original (lr 3e-4 annealed, tanh 32-32, Exp 1/5 PPO defaults) and fixes
the unpublished ones: both sides of self-play games are training data, the opponent is the current policy in half of
the games and one of the last 16 snapshots (one per 100 iterations) otherwise, gamma 1. REINFORCE uses the same data
(gamma does not apply: return = final outcome) with one Adam step per iteration and a batch-mean baseline.

| setting | value | source |
|---|---|---|
| network | actor 243-32-32-82 tanh, critic 243-32-32-1 tanh (separate), orthogonal init (hidden sqrt 2, actor out 0.01, critic out 1) | original (Exp 1 meta) + inference |
| schedule | 20,000 iterations x 1,024 full games (all played to the end, <= 162 plies) | original |
| PPO | Adam lr 3e-4 (eps 1e-5) linearly annealed to 0, grad norm 0.5, 4 epochs x 8 minibatches (of games), clip 0.2 (policy and value), entropy 0.01 (over legal moves), value 0.5, GAE lambda 0.95 per player | original defaults |
| gamma | 1.0 | search (v4) |
| training data | both players' moves in games against the current policy; the learner's moves in games against snapshots | search (v2) |
| opponents | current policy with probability 0.5, else uniform over the last 16 snapshots (one every 100 iterations) | search (v2) |
| REINFORCE | same rollouts; one Adam step (lr 3e-4 annealed) per iteration on all trained moves, return = final outcome, batch-mean baseline, entropy 0.01 | ours |
| learner colour | Black or White at random per game | ours |
| reward | +1 / -1 at the end (pgx: area score, komi 7.5; superko repeat loses) | game rule |
| checkpoints | `state.pkl` every 250 iterations (exact resume), `iter<k>.npz` every 1,000 | ours |

Archived: the first (v1) attempt in `~/wsp/exp6_data/opponents_v1/` (checkpoints up to iteration 7,000). All search
runs: `~/wsp/exp6_data/train/<variant>/<name>/` (config.json, train/eval logs, calib_<iter>.json).

### Final opponents

Trained with `scripts/final_chain.sh` (REINFORCE screen, then `scripts/train_final.sh v4 v4`), 2026-11-06 02:33-05:13
server time; `ppo_s1` is the screened v4 run itself (same seed and schedule, continued exactly). Calibration with 2,000
games per reference (`score.py opponents` -> `~/wsp/exp6_data/opponents/strength.json`, `strength.md` with each
seed against its own seed's targets and the standard errors). Win rates in %; "mean diff" over the 7 reference
networks; standard error of one cell about 1.0-1.1 points (ours) and 1.1 (original, 2,000 games) or 2.2 (original
"strong", 500 games), i.e. about 1.5-2.5 points for a difference.

| ours (final) | Astra b0 | Astra b10 | Astra b25 | Astra b50 | PPO-512 s0 | PPO-512 s1 | PPO-512 s2 | random | mean diff vs own seed | vs family |
|---|---|---|---|---|---|---|---|---|---|---|
| ppo_s0 | 66.0 | 41.0 | 47.1 | 19.5 | 38.4 | 35.0 | 41.7 | 99.5 | +8.1 | +6.7 |
| ppo_s1 | 61.8 | 37.2 | 35.6 | 15.5 | 44.2 | 24.3 | 49.5 | 99.5 | +5.3 | +3.8 |
| ppo_s2 | 58.1 | 33.6 | 41.2 | 24.8 | 20.6 | 54.9 | 54.3 | 99.5 | +3.5 | +6.5 |
| PPO mean of our seeds | 62.0 | 37.3 | 41.3 | 19.9 | 34.4 | 38.0 | 48.5 | | +5.7 |
| PPO family diff per ref | +14.0 | +6.9 | +12.3 | +2.8 | -7.6 | +8.8 | +2.5 | | | |
| reinforce_s0 | 1.2 | 1.2 | 4.0 | 0.2 | 7.3 | 6.9 | 7.9 | 65.2 | -1.0 | +0.5 |
| reinforce_s1 | 1.8 | 1.1 | 2.3 | 0.0 | 9.8 | 9.4 | 10.8 | 91.1 | +0.0 | +1.5 |
| reinforce_s2 | 1.4 | 1.0 | 2.4 | 0.2 | 6.3 | 6.0 | 8.9 | 93.4 | +1.7 | +0.2 |
| REINFORCE mean of our seeds | 1.4 | 1.1 | 2.9 | 0.1 | 7.8 | 7.4 | 9.2 | | +0.7 |
| REINFORCE family diff per ref | +0.7 | +0.6 | +1.0 | +0.0 | -0.4 | +1.7 | +1.5 | | | |

Against the prompt's vs-random statements: "strong" 99.5 % (stated 98 %), "weak" 91.1 % (stated 95 %); hidden
opponents 65.2-99.5 % (stated 79-98 %): **reinforce_s0 at 65.2 % is below the stated range** (its calibration against
the 7 reference networks is fine, -1.0 vs its own seed's targets).

**What matches and what does not.**
- REINFORCE: within 2 points of the original REINFORCE family on every reference (mean +0.7); each seed within
  2 points of its own seed's targets on average.
- PPO: +5.7 points stronger than the original PPO family on average (seeds +3.8 / +6.7 / +6.5). Against the
  same-size PPO networks the family is matched on average (-7.6 / +8.8 / +2.5, i.e. within the spread between the
  original seeds themselves), but **against Astra's hand-written networks our PPO opponents are clearly stronger**
  (+14.0 vs budget 0, +12.3 vs budget 25, +6.9 vs budget 10, +2.8 vs budget 50). That bias is not a pure strength
  offset (a weaker recipe would push the same-size results below target; v2, v6, v7 show the same pattern), so our
  opponents exploit Astra's style of network more than the original ones did. Individual references per seed differ
  by up to 30 points from that seed's own target; the original seeds differ from each other by up to 26 points on the
  same reference, so per-seed, per-reference agreement within 5 points is not achievable by any recipe.
- Consequence: absolute win rates against our hidden opponents are lower than the paper's for hand-written networks.
  Astra's released networks against our opponents (`score.py astra`, `~/wsp/runs/astra-rescored/exp6/scores.json`):

| Astra network | PPO s0 / s2 ours | PPO s0 / s2 paper | main (4 trained) ours | main paper | h2h same-size PPO s0 / s1 / s2 (ours; paper for b0, b50) |
|---|---|---|---|---|---|
| budget 0 | 34.7 / 40.3 | 54.1 / 54.9 | 68.1 | 76.9 | 43.0 / 26.4 / 35.6 (42.6 / 26.3 / 36.3) |
| budget 10 | 60.4 / 65.2 | 72.9 / 60.0 | 80.8 | 82.9 | 43.1 / 45.8 / 57.3 |
| budget 25 | 51.6 / 57.8 | 76.0 / 62.3 | 76.0 | 83.6 | 46.9 / 43.8 / 50.7 |
| budget 50 | 81.1 / 75.7 | 84.0 / 80.8 | 89.1 | 91.1 | 72.8 / 67.0 / 80.3 (74.2 / 70.0 / 78.5) |

  The ordering of Astra's budgets is kept (0 < 25 < 10 < 50 on our PPO opponents; paper 0 < 10 < 25 < 50 on PPO
  s0), but budget 0 loses about 17 points. Opus must be compared with these re-scored Astra numbers (same opponents,
  same scorer), and the h2h against the released same-size PPO networks, which does not depend on our opponents.

**GPU time** (lock-holding time, `~/wsp/exp6_data/logs/gpu_time.jsonl`, plus v1 and benchmarks): v1 0.3 h, screening
and search v2-v8 2.6 h, final six networks 3.3 h (incl. ppo_s1 continued from the screen), benchmarks, calibration
and re-scoring about 0.3 h: **about 6.5 GPU hours** of the 24-hour budget.

### Integrity (non-ECC RAM)

- Opponent files are written with verified writes (write, fsync, drop the cached pages, re-read and compare SHA-256)
  and listed in `~/wsp/exp6_data/opponents/manifest.sha256` (`<sha256>  <path relative to opponents/>`, hashes of the
  disk copies; also checked by `~/wsp/harness/check_inputs.py exp6`).
- `run.py` verifies the practice opponents against the manifest when a conversation starts and before every games
  pass; `passtool.py` and `score.py` verify each opponent folder again right before loading it. On a mismatch the
  cached pages are dropped and the file is checked once more; if it is still wrong the run stops with an
  `IntegrityError` (logged to `~/wsp/exp6_data/logs/integrity.log`), nothing is shown to the model and the turn is
  handled again on restart.
- Released networks (Astra's, same-size PPO) are checked against the logs repo's git blob hashes before loading.
- The model's final network: copied from the sandbox folder (flushed and read from disk) with verified writes to
  `<run>/network/`, hashed into `<run>/network/sha256.txt` and `state.json` (`extra.network_sha256`). The scorer
  re-verifies it before scoring and writes every file it loaded with its SHA-256 to `<run>/eval/inputs.json`, so a
  later re-score can confirm nothing changed.
- Tested: a flipped bit in a copied opponent file is detected and reported; mock runs and the scorer pass the checks
  (`inputs.json` lists 95 verified files per scored network).

