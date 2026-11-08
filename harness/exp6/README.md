# Experiment 6 harness: write a 1.5M-weight 9x9 Go network from scratch

Replication of `lm_understand-logs/exp6_go_from_scratch` (gpt-6-astra, Codex CLI, effort `xhigh`) with Claude
through the Claude Code CLI. Four conversations: pass budgets 0, 10, 25 and 50.

## How to run

```
python3 ~/wsp/harness/exp6/run.py [--model claude-opus-5-5] [--effort xhigh] [--plan opus-5.5_paper-effort] \
                                  [--only budget_0,budget_10,...] [--mock] [--no-score]
```

- Runs the conversations one after another, then scores them (`score.py`, under `~/wsp/gpu.lock`) and writes
  `~/wsp/runs/<plan>/exp6/scores.json` and `summary.md`. Each conversation is in `~/wsp/runs/<plan>/exp6/budget_<B>/`.
- Safe to re-run: each conversation resumes from its `state.json` (core `Conversation`); a reply that was saved but not
  yet handled (harness stopped between the model call and the tool run) is handled on restart instead of re-sending
  the previous message. Finished conversations are only re-scored.
- `--mock` replays the original model's replies (`runs/budget_<B>/turns/*/answer.md`) instead of calling Claude;
  output goes to `~/wsp/runs/_validation/exp6/mock/`.
- **Prerequisite for budgets 10-50 and for scoring:** the trained opponents in `~/wsp/exp6_data/opponents/`
  (`ppo_s0..2`, `reinforce_s0..2`; see "Opponents"). Budget 0 needs no opponent until scoring.
- Astra's released networks against our opponents (do once, after training): 
  `flock ~/wsp/gpu.lock ~/wsp/venvs/jaxenv/bin/python ~/wsp/harness/exp6/score.py astra`
  (-> `~/wsp/runs/astra-rescored/exp6/`; `summary.md` of a run picks it up).

## Files

| file | runs in | what it does |
|---|---|---|
| `run.py` | system python3 | the driver: messages, block parsing, limits, budget accounting, reminders, sandbox runs (core `sandbox.run_python` with the original PRELUDE), passes (subprocess), copying the final network, scoring |
| `passtool.py` | jaxenv (CPU) | the games pass and the positions pass: validates the request, writes `/work/pass_<k>/`, returns the result shown to the model |
| `gocore.py` | jaxenv | the Go engine (pgx `go_9x9` rules), 243-input encoding, MLP policies, batched sampled games, the per-opponent summary, position replay |
| `score.py` | jaxenv (GPU) | the hidden evaluation, head-to-heads against the released same-size PPO networks, Astra re-scoring, h2h reproduction |
| `train_opponents.py` | jaxenv (GPU) | PPO / REINFORCE self-play training of the six small opponents, resumable, benchmark mode |
| `calibration.py` | jaxenv | calibration of our opponents against the original ones from released data only (targets, `calibrate`, `compare`) |
| `integrity.py` | stdlib | non-ECC RAM safeguards: verified writes, the opponents' `manifest.sha256`, verify-before-load with one cache drop and re-check, network `sha256.txt`, git-blob check of released files |
| `scripts/screen.sh` | | trains the first N iterations of recipe variants (screening; continuable exactly) |
| `scripts/final_chain.sh` | | the chain actually run: REINFORCE screen, then `train_final.sh v4 v4` |
| `scripts/train_final.sh` | | the six final opponents with the chosen recipe, exported + manifest, then `check_inputs.py exp6`, `score.py opponents` and `score.py astra` |
| `scripts/screen_table.py`, `scripts/calib_ckpt.py` | | the variant table below; calibration of any checkpoint |
| `scripts/bench.sh` | | the training benchmark |
| `scripts/mock_validate.sh`, `scripts/make_placeholders.py` | | validation only: mock replays with untrained placeholder opponents (`EXP6_OPPONENTS=~/wsp/runs/_validation/exp6/placeholder_opponents`) |
| `scripts/compare_mock.py`, `compare_networks.py`, `compare_positions.py`, `test_protocol.py` | | validation only: turn-by-turn comparison with the original, final networks and positions-pass files vs the released ones, synthetic replies for the protocol paths Astra never used |

Data on the server: `~/wsp/exp6_data/` (`train/<name>/` checkpoints and logs, `opponents/<name>/` the exported
opponents, `logs/`, `jax_cache/`).

## What is copied from the original, and from where

| item | source in `lm_understand-logs` | here |
|---|---|---|
| first message (per budget) | `exp6_go_from_scratch/runs/budget_<B>/message.txt` | read verbatim at run time (sha256 stored in `state.json`); the mock turn-1 and turn-2 prompts are byte-identical |
| system prompt, no tools | `harness/instructions.md`, `harness/codex_config.toml` | core `claude_turn.py` |
| sandbox and PRELUDE (`SHAPES`, `NAMES`, `load`, `save`, `saved`, `inp`, `mv`, `PASS`, `/work/arch.json` = `{"depth": 6, "width": 512}`) | `harness/sandbox.py` | PRELUDE verbatim in `run.py`; sandbox = core `sandbox.py` |
| limits: 120 Python runs, 900 s per run, output cut at 20,000 characters | `message.txt`, `harness/sandbox.py` | same |
| Python result message `OUTPUT of Python run <n> of at most 120:` + fenced output | `turns/t*/prompt.txt` | same (byte-identical in the mock replay where the output does not depend on a pass) |
| games/positions result message `<Kind> pass <k> of <B> done.` + fenced JSON + `Passes left: <n>.` | `turns/t*/prompt.txt`, `state.json` | same, same JSON keys and order |
| games pass summary per opponent (score, wins, losses, score_as_black/white, mean_margin, mean_length, stones_captured/lost_per_game, passes_per_game, games_with_a_pass_before_move_60, end_two_passes/repetition/move_limit), opponents in name order, float32 rates | `turns/*/prompt.txt`, `work/pass_<k>/summary.json`, `eval/*/eval_submission.json` | `gocore.summary` |
| `/work/pass_<k>/README.txt` texts (games; positions with / without activations) | `work/pass_<k>/README.txt` | verbatim |
| `games.json` record: `game, opponent, net_color ("black"/"white"), result, margin, end ("two passes"/"repetition"/"move limit"), stones_captured_by_net, stones_lost_by_net, moves: [{ply, by ("net"/"opponent"), move, top5 [[move, p], ...]}]` | not released; recovered from Astra's own printouts of the file (`runs/budget_{10,25,50}/tool_calls.jsonl`) | same keys and values; probabilities rounded to 4 decimals as printed there |
| `boards.npy` (games, 162, 81) int8 board after each ply, `valid.npy` (games, 162) | README text | same |
| positions pass files `inputs.npy, legal.npy, logits.npy, probs.npy, act_L1..6.npy` (float32 / bool), top-3 as `[["(r,c)" or "pass", p rounded to 4], ...]` | `runs/budget_{25,50}/work/pass_*` | same |
| requests saved as `requests/<kind>_<k>.json` (indent 1) | `runs/budget_*/requests/` | same |
| `tool_calls.jsonl`: python entries `{t, kind, s, code, out}`, pass entries `{t, kind, k, request, result}` | `tool_calls.jsonl` | same (python `out` not truncated to 6,000 characters) |
| hidden evaluation: 2,000 games per opponent, opponents `ppo_s0, ppo_s2, reinforce_s0, reinforce_s2, random`, seed 777 + j, network Black in half, main = mean of the four trained | `eval/ppo_same_size/ppo_big_eval/eval_ppo_big.py`, `eval_lr_scan.py`, `eval/budget_*/eval_submission.json` | `score.py` |
| head-to-head: 2,000 games, seed 31337, written network Black in half, vs `ppo_same_size_lr3e-5_s{0,1,2}` | `h2h_lr_scan.py`, `h2h_lr_scan.json` | `score.py` |
| analysis vs "strong": 500 games, seed 4242 (summary only; the failure statistics are not reproduced) | `eval/budget_*/analysis_strong.json` | `score.py` |

Seeds are the original's, but JAX random streams are not theirs, so individual games differ; only the statistics are
comparable.

## Protocol details (inferred where the logs do not show them)

The released runs contain no reminder, no invalid request, no reply with several blocks and no reply after the limits
(every turn is exactly one block or the final answer: turns = 1 + runs + passes). The original harness loop was not
released. So these messages are **inferred**; the wording is ours, modelled on the same authors' Exp 7/8 harness
messages ("The forward request is not valid (JSONDecodeError: ...). Not counted.", "You have used k of N runs. ..."):

- Final answer with passes left (budget > 0): `You have used <p> of <B> passes. The session continues until every pass
  is used: use the remaining <B-p> before you give your final answer.` (the README says "a final answer with passes
  left gets a reminder"; counted in `state.json` `extra.reminders`).
- Invalid request (bad JSON, unknown opponent, odd count, more than 128 games / 256 positions, illegal or post-game
  move in a `moves` position, malformed board): `The <kind> request is not valid (<error>). Not counted.\nPasses
  left: <n>.` Not counted as a pass ("a request that fails is not counted"); the request is kept as
  `requests/<kind>_<k>_invalid_<n>.json`.
- A reply with more than one runnable block: nothing is run, `Your reply had <n> blocks, so nothing was run (nothing
  was used). Reply with exactly one block (```python, ```games, ```positions), or without one to finish.`
- A block after the limit: `No Python runs left (120 of 120 used); the block was not run.` /
  `No passes left (<B> of <B> used); the block was not run.`
- A reply is the final answer when it has no ```python block (budget 0) / no ```python, ```games or ```positions
  block (budgets > 0) and no passes are left. Text around a single block is allowed (Astra never wrote any). Blocks of
  other languages are ignored. Budget 0 has no pass tool at all: a ```games block there is not a tool.
- A failing or timed-out Python run counts as a run (as in the original sandbox, which returns the error output).
- If the pass tool itself crashes (a harness problem, not the request), it is retried 3 times and then the driver
  stops with an error; nothing is shown to the model and the turn is handled again on restart.
- Which games the network plays Black in: the first half of the games against each opponent (in the released
  records game 0 is Black against "random").
- `passes_per_game` and `games_with_a_pass_before_move_60` count the network's passes (checked: 5.09 passes per game =
  50.5 network moves x 0.1008 pass rate in `eval/budget_0/analysis_strong.json`), "before move 60" = ply index < 60.
- Pass seeds: games pass k, j-th opponent in name order: seed 1,000,003 k + 7,919 j (ours).

## Rules check: pgx 2.6.0 `go_9x9` against the prompt

`pgx.make("go_9x9")` = `Go(size=9, komi=7.5)`; we use its `Game` class directly (`history_length=1`; our 243-number
input equals pgx's observation planes: mover, opponent, colour).

| prompt says | pgx 2.6.0 | match |
|---|---|---|
| captures; suicide illegal; simple ko illegal | `legal_action_mask`: empty and (adjacent empty, capture, or own group with >1 liberty); ko point masked | yes |
| repeating an earlier whole-board position ends the game, mover loses | Zobrist hash of the board after each ply; a non-pass move whose hash was seen before sets `is_psk`, terminal, the mover gets -1 | yes (positional; hash collisions negligible) |
| ends after two consecutive passes or 162 moves in total | `consecutive_pass_count >= 2`; `max_termination_steps = 9*9*2 = 162` counts passes too | yes |
| area scoring, komi 7.5, no draws | stones + empty points not reachable from the opponent's stones; Black wins if B - 7.5 > W | yes, one edge case: an empty region touching no stone at all counts for both players in pgx (prompt: for neither). Only possible on an empty board; the result is the same (White wins on komi) |
| illegal moves removed, sampled at temperature 1 | (harness) logits of illegal moves set to -inf, `jax.random.categorical` | yes |

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

## Validation

All validation outputs are in `~/wsp/runs/_validation/exp6/` (mock runs in `mock/`). No real model call was made.

**Mock replays of Astra's four conversations** (`scripts/mock_validate.sh`, untrained placeholder opponents):

| run | turns (orig / ours) | prompts byte-identical | other prompts: same header line | pass results with identical JSON key structure and `Passes left` line | tracebacks in Python outputs (orig / ours) | final network vs `networks/budget_<B>` |
|---|---|---|---|---|---|---|
| budget_0 | 2 / 2 | 2 (turn 1 = message.txt, turn 2 = the Python output) | - | - | 0 / 0 | 14/14 tensors bit-identical |
| budget_10 | 28 / 28 | 11 | 17 | 10 of 10 | 0 / 0 | 14/14 bit-identical |
| budget_25 | 56 / 56 | 25 | 31 | 25 of 25 | 1 / 1 | 14/14 bit-identical |
| budget_50 | 90 / 90 | 32 | 58 | 49 of 49 differing (1 positions result byte-identical) | 0 / 0 | 14/14 bit-identical |

Prompts differ only where they report games against our (placeholder) opponents or Python output computed from
those games. Astra's analysis code read our `games.json` / `boards.npy` without a single error. The networks Astra's
code builds do not depend on the pass numbers, so all four final networks are reproduced bit for bit
(`network_comparison.json`, `mock_comparison.json`).

**Positions passes** (deterministic, so directly comparable; `positions_comparison.json`): budget_25 pass 23: all 11
files bit-identical (inputs, legal, logits, probs, act_L1..6, README). budget_50 passes 2, 36, 37: inputs, legal and
README identical; logits differ by at most 3.1e-5 and probs by at most 9.2e-12 (float32 summation order of a
different BLAS); the top-3 lists equal up to the order of exactly tied probabilities.

**Head-to-head reproduction** (`score.py validate_h2h`, `h2h_reproduction.json`): the released Astra networks
against the released same-size PPO networks, 2,000 games each, our engine vs the released `h2h_lr_scan.json`:

| pair | released | ours | z |
|---|---|---|---|
| budget_0 vs PPO lr3e-5 s0 | 42.55 % | 42.95 % | +0.26 |
| budget_0 vs s1 | 26.30 % | 26.35 % | +0.04 |
| budget_0 vs s2 | 36.25 % | 35.55 % | -0.46 |
| budget_50 vs s0 | 74.15 % | 72.80 % | -0.97 |
| budget_50 vs s1 | 69.95 % | 67.05 % | -1.97 |
| budget_50 vs s2 | 78.55 % | 80.30 % | +1.37 |

All within sampling error (sum of z^2 = 7.0 on 6 degrees of freedom). The other summary fields also agree (e.g.
budget_0 vs s0: margin 11.4 vs 11.3, length 92.8 vs 92.6, captured 24.9 vs 24.3, lost 7.6 vs 7.6, network passes
per game 5.24 vs 5.12, ends 1968/30/2 vs 1977/23/0). This checks the input encoding, the rules, masking, sampling and
the summary definitions against the authors' own engine.

**Protocol paths Astra never hit** (`scripts/test_protocol.py`, budget 2, synthetic replies): bad JSON, odd count,
>128 games, unknown opponent, illegal move in a `moves` position -> "not valid ... Not counted", budget unchanged;
two blocks -> nothing run; final answer with passes left -> reminder; valid positions pass (moves + board, with
activations) and games pass (all three opponents) -> counted; a games block after the budget -> refused; a Python
run -> `OUTPUT of Python run 1 of at most 120` (PRELUDE works: `inp(1,2,0)` = 33). State afterwards: passes 2,
requests 7, invalid 5, reminders 1, several_blocks 1.

**Scorer end to end** (`score.py runs` on the four mock runs, CPU, placeholder opponents): writes `eval/` per run,
`scores.json`, `summary.md`; ~2 min per network on 3 CPU cores (seconds on the GPU). The h2h columns for the
replayed budget 0 / 50 networks: 43.0 / 26.4 / 35.6 and 72.8 / 67.0 / 80.3 (paper: 42.6 / 26.3 / 36.3 and
74.2 / 70.0 / 78.5).

**Other checks:** a games pass of 128 games takes 9-13 s on the CPU including JIT compilation (passes run on the
CPU so they never wait for the shared GPU); the trainer resumes exactly from `state.pkl` (killed at iteration 250+,
restarted, "resumed at iteration 250"); `run.py` refuses to start a budget > 0 conversation before the practice
opponents exist.

## Deviations

| # | deviation | reason | effect |
|---|---|---|---|
| 1 | Opponents (practice "weak"/"strong", hidden PPO s0/s2, REINFORCE s0/s2) retrained by us; unpublished settings chosen by a calibration search on released data (recipe v4, see "Recipe search") | not released | Absolute win rates against the hidden opponents are **not comparable with the paper**. Comparable: (a) Opus vs Astra's released networks against the same (our) opponents with the same scorer (`score.py astra`); (b) the head-to-head against the released same-size PPO networks, which does not depend on our opponents at all. The pass results the model sees also differ from Astra's. |
| 2 | Reminder, invalid-request, several-blocks and out-of-limit messages are our wording | never occurred in the released runs; harness loop not released | only matters if Claude triggers them; counted in `state.json` `extra` |
| 3 | Sandbox: 16 GB and 6 threads per run instead of 96 GB / 8 threads | server (core `sandbox.py`) | none seen in the replays (Astra's longest run took 11 s) |
| 4 | Passes run on JAX CPU, scoring on the GPU; random streams are not the original's | shared GPU | game-level results differ, distributions do not (h2h check) |
| 5 | `state.json` `extra.requests` counts invalid requests too (the original's `requests` always equalled `passes`) | bookkeeping | none |
| 6 | `analysis_strong.json`: summary of the 500 games only, not the failure statistics | not part of the post | none for the post's numbers |
| 7 | Training resumes exactly (Adam state kept) instead of the original's resume with Adam reset | better reproducibility | none |
| 10 | Our PPO opponents are +5.7 points stronger than the original ones on average (+12 to +14 against Astra's budget 0 / 25 networks); reinforce_s0 beats random 65 % (prompt: hidden 79-98 %) | best recipe found (see "Final opponents") | absolute hidden scores of hand-written networks lower than the paper's; compare Opus with the re-scored Astra numbers |
| 8 | Python entries in `tool_calls.jsonl` keep the full output (original: first 6,000 characters) | core sandbox | none |
| 9 | pgx counts an empty region bordering no stone for both players (prompt: for neither) | pgx rule; same for the original, which also used pgx | only an empty board; the winner is the same |

## Open questions

1. Our PPO opponents are about 6 points stronger than the original ones on average, mostly against Astra's
   hand-written networks (+12 to +14 vs budgets 0 and 25). Opus vs Astra stays comparable (both scored against the
   same opponents), but absolute numbers are not the paper's. A further search would need a recipe change that makes
   the opponents less exploitative of hand-written networks specifically, which the released data can only measure
   on these 7 references (risk of over-fitting them); not attempted.
2. reinforce_s0 beats the random player only 65 % of the time; the prompt (kept verbatim) says the hidden opponents
   beat it 79-98 %. "Weak" is at 91 % (stated 95 %), "strong" at 99.5 % (stated 98 %).
3. The reminder / invalid-request wording is inferred (see above); confirm with the authors if possible.
