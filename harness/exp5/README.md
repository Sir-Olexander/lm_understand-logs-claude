# Exp 5 replication: write MinAtar networks from scratch (Breakout, Seaquest, Asterix)

Replicates `exp5_write_from_scratch/` of the original logs (server `~/wsp/lm_understand-logs/`): one fresh
conversation per game, one message, one reply, no tools; the reply's last `edits` block sets every weight of a
600-32-32-6 tanh network (all weights start at 0); the network is scored on 1,000 MinAtar 1.0.15 episodes.
Paper effort: `xhigh`.

## How to run (on the server)

```
python3 ~/wsp/harness/exp5/run.py                      # real run: claude-opus-5-5, effort xhigh, plan opus-5.5_paper-effort
python3 ~/wsp/harness/exp5/run.py --only breakout      # some games only (comma-separated)
python3 ~/wsp/harness/exp5/run.py --model M --effort E --plan P
python3 ~/wsp/harness/exp5/run.py --mock               # replay the original gpt-6-astra replies (no Claude call)
```
Output: `~/wsp/runs/<plan>/exp5/<game>/` (or `~/wsp/runs/_validation/exp5/<game>/` with `--mock`), holding the core
conversation files (`message.txt`, `turns/t001/...`, `state.json`, `final_answer.md`, `loop.log`) plus `hand.npz`
(the network, same keys as the original), `apply.json` (how the reply was read; same fields as the original's) and
`eval.json` (the 1,000-episode score with per-episode returns and lengths). `scores.json` and `summary.md` are in
`<plan>/exp5/`. Re-running is safe: a game with `final_answer.md` is not asked again; the reply is re-read (fast,
deterministic); the score is reused if the network's sha256, episode count and seed are unchanged.
The harness runs with the system `python3` (stdlib only); reading and scoring run in `~/wsp/venvs/jaxenv` (numpy,
minatar 1.0.15) on the CPU with 3 worker processes (1 thread each). No GPU, so no `gpu.lock`. Scoring takes
~35 s (Breakout), ~10 s (Seaquest), ~7 s (Asterix). Nothing was installed.

## Files

| File | What it does |
|---|---|
| `run.py` | Entry script: one-shot conversation per game (`core/convo.one_shot`), read reply, score, `scores.json`, `summary.md`. |
| `prompts/<game>.txt` | The first message, a byte-identical copy of the original `runs/<game>/message.txt` (= `reply/prompt.txt`; the original README and `cmp` confirm they are the same). sha256: breakout `766ee251…`, seaquest `81e51857…`, asterix `9e8243d5…`; 9,973 / 18,266 / 10,930 characters, as in the original `loop.log`. |
| `edits.py` | The edits-language reader (the prompt's "HOW TO WRITE THE WEIGHTS"). |
| `apply_reply.py` | Reply -> `hand.npz` + `apply.json` (optionally compares with a reference network). |
| `score_minatar.py` | The MinAtar scorer. |
| `games.py` | Channel mappings (copied from the prompts, checked against the prompt text at every run), npz keys, original numbers. |
| `check_parse.py` | Validation: reads the three original replies and compares with the original `hand.npz` and `apply.json`. |
| `test_edits.py` | Edge cases of the reader (prompt examples, multi-line lists, skipped statements, last block, cut-off, no block). |

## What is copied from the original, and how

- **Message**: verbatim (see above). No system-prompt changes beyond core's neutral line (original:
  `instructions.md` = "Answer the user's request.").
- **Reading the reply** (`edits.py`), per the prompt: the LAST block whose opening line is ```` ```edits ````; one
  statement per line; `#` comments; a statement continues over the next lines while its `[` brackets are open;
  `T[i, j] = v`, `= [list]`, `= [[2-D list]]`, `*=`, `+=`, `-=`; indices integer, `a:b`, `a:b:s`, `:`; statements
  applied in order; an unreadable statement is skipped and the others apply; values plain decimals with |v| <= 1e6.
  The original's first reader rejected `[ [ ...` (a 2-D list whose first row starts after a space/newline), which
  hit 6 Seaquest and 4 Asterix statements (`run.log`); the original then fixed its reader and re-read the same
  replies (`apply.json`, 0 errors), and those networks are the scored ones. We implement the fixed reader.
  `apply.json -> edit_report` has the original's fields (`block, statements, applied, entries_written, errors,
  cut_off, tensors_touched`) plus `nonzero`, `nonzero_total`, `parameters`, `max_abs`.
- **Scoring** (`score_minatar.py`), per the prompt and README: `minatar.Environment(game, sticky_action_prob=0.1)`
  (MinAtar 1.0.15, difficulty ramping on = package default); the installed game files are identical to the source
  printed in the prompts (checked with `diff`, only the prompt's trailing blank line differs). Input channel k =
  OR of the game's `state()` channels listed in the prompt ("How the 6 channels are built ..."); input index
  `(row*10+col)*6+channel`. Network `tanh(x W1 + b1)`, `tanh(· W2 + b2)`, `· W3 + b3` in float32; action sampled from
  the softmax; logit index = MinAtar action index (0 n, 1 l, 2 u, 3 r, 4 d, 5 f), which is what the prompt states
  ("act(a) with a = 0..5 meaning 'n','l','u','r','d','f'"). Score = sum of rewards; episode ends at game over or
  after 10,000 steps; 1,000 episodes; reported mean, SE, max, mean length.

## Choices where the original is not explicit (inferred / defined here)

- **Random streams.** The original's seeds are not in the logs, so episodes cannot be replayed exactly. Ours:
  episode e gets a fresh `Environment`, `env.seed(seed*1_000_003 + e)` then `reset()` (MinAtar's one RandomState
  drives both the game and the sticky-action draw, as in the package), and its own action sampler
  `numpy.random.default_rng([seed, e])`; default seed 0. Results do not depend on the worker split.
- **SE** = population std / sqrt(n) (the original's ddof is unknown; the difference is a factor 0.9995).
- **Reader details the prompt leaves open:** numpy index semantics (negative indices allowed, slices clipped to the
  axis; an out-of-range integer index is an error); a list must match the block's shape (singleton axes ignored);
  a line that starts a new statement ends an unclosed list (the unclosed one fails) so one bad bracket cannot eat
  the rest of the block; no closing fence -> the rest of the reply is the block and `cut_off` is true; no edits
  block -> all weights stay 0 and that all-zero network (uniform random policy) is scored; float64 arithmetic,
  stored as float32; `ZERO <tensor>` (from the same harness's Exp 4 language) is accepted though not advertised in
  the Exp 5 prompt. With the three original replies none of these choices matters (all statements read, 0 errors).
- **Answer text.** The original took Codex's last agent message (`-o answer.md`); here it is the final result text
  of `claude -p` (core). Only the last `edits` block is read in both cases.

## Validation (mock mode, `python3 run.py --mock`, outputs in `~/wsp/runs/_validation/exp5/`)

**Reading:** the three original replies read with `edits.py` give networks bit-identical (float32, every tensor) to
the original `networks/<game>/hand.npz`, and every `edit_report` field and nonzero count equals the original
`apply.json`: Breakout 92/92 statements, 857 entries written, 815 nonzero; Seaquest 122/122, 905, 803; Asterix
55/55, 1,065, 637 (`check_parse.py`: ALL OK).

**Scoring** the original networks (1,000 episodes, seed 0 = what `run.py --mock` reports), and pooled over seeds 0-5
(6,000 episodes, `_validation/exp5/extra_seeds/`):

| Game | Original (`eval/minatar_eval.json`) | Ours, seed 0 | Ours, seeds 0-5 pooled | Difference / combined SE (pooled) |
|---|---|---|---|---|
| Breakout | 517.4 ± 7.4 (len 7,417) | 502.9 ± 7.7 (len 7,188) | 505.7 ± 3.1 | -11.7, z = -1.5 |
| Seaquest | 53.1 ± 1.4 (len 1,412) | 52.6 ± 1.4 (len 1,396) | 53.4 ± 0.6 | +0.3, z = 0.2 |
| Asterix | 53.2 ± 1.0 (len 858) | 53.3 ± 1.0 (len 861) | 54.1 ± 0.4 | +0.9, z = 0.8 |

Seaquest and Asterix agree within one SE. Breakout's seed-0 mean is 14.5 below the original (1.35 combined SE;
per-seed means 499-513); pooled it is 1.5 combined SE below. No systematic cause was found: the forward pass in
float64 gives the same 1,000 episodes as float32; the source of the games equals the prompt's; score per step is the
same (0.0700 vs 0.0698), the difference is in episode length (about half the Breakout episodes reach the
10,000-step cap). We treat it as sampling noise but note it.

## Deviations from the original

- Model/CLI: Claude via `claude -p` (core) instead of `codex exec`; the original requested reasoning summaries
  ("detailed"); core requests summarized thinking. Tool audit is core's.
- Random seeds differ (unknown in the original), see above.
- The PPO baselines in the original `eval/` are not re-scored (their checkpoints are not released).
