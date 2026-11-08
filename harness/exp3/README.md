# Experiment 3: which language a language model learned, from its middle blocks (replication harness)

Replicates `lm_understand-logs/exp3_language`. Three character-level transformers were trained on Python, romaji and
English. For each, the model gets only the four transformer blocks, as a dict `W` inside a Python sandbox. It must say which corpus the network was trained on
and label each of the 24 heads. No forward pass is allowed. There are 6 conversations, two per network, and each is
multi-turn: one code block per turn, up to 100 runs. Paper effort: `xhigh`.

## How to run (on the server)

```
python3 ~/wsp/harness/exp3/run.py                      # real runs: claude-opus-5-5, effort xhigh, all 6 conversations
python3 ~/wsp/harness/exp3/run.py --model <id> --effort <level> --plan <plan name> --only 7d252c62_r0 --threads 6
python3 ~/wsp/harness/exp3/run.py --mock --threads 3   # validation: replays Astra's replies, its code runs in OUR sandbox
python3 ~/wsp/harness/exp3/score.py --original         # scorer on Astra's original final answers vs eval/scores.json
python3 ~/wsp/harness/exp3/score.py <out_root>         # re-score a finished plan folder
python3 ~/wsp/harness/exp3/replay_compare.py <out_root>  # compare a mock replay's outputs with the original
```

Outputs:
- Real runs go to `~/wsp/runs/<plan>/exp3/<run id>/`, using the core layout: `message.txt`, `turns/tNNN/`,
  `tool_calls.jsonl` (full outputs), `state.json`, `final_answer.md`, `loop.log`, and `work/run_NNN/` (each run's
  `/work`).
- After the runs, the script writes `~/wsp/runs/<plan>/exp3/scores.json` and `summary.md`.
- `config.json` records the model, effort, threads, limits and the sandbox prelude.
- `--mock` writes to `~/wsp/runs/_validation/exp3/`. This folder is wiped first. The script also writes `validation.json`
  there. A full replay takes about 70 s.
- **Re-running is safe:**
  - Finished conversations are skipped.
  - Unfinished ones continue from `state.json`.
  - `recover()` also handles a harness killed mid-turn. It restores core's transcript backup, so a half-done turn is
    not left in the Claude session. It also processes a reply that was saved but whose code never ran, so the
    previous message is not sent again. This was tested with a fake CLI.
- Conversations run one after another: all `_r0`, then all `_r1`. The original ran all six in parallel.

**Run IDs are neutral.** They have the form `<network id>_r0` / `_r1`. The network IDs are random hex: `7d252c62`,
`9e92cad6` and `4ba2ba90`. The mapping to language and to the original run lives only in `eval_key.json`, which the
model cannot see. The weights are copied to a neutral path, `<out_root>/_weights/<network id>.npz`, checked by SHA-256,
and the sandbox bind-mounts only that copy, at `/data/blocks.npz`. So even `/proc/self/mountinfo` inside the sandbox
shows no path that names a language. Nothing else from the logs repo is mounted.

## Files

| File | What it does |
|---|---|
| `run.py` | Driver: the conversation handler (code block → sandbox → next message), crash recovery, weight staging, scoring, and `--mock` validation. |
| `score.py` | Scorer (stdlib). Parses `corpus:`, `order:` and `probabilities:`, plus the 24 head labels of the `Heads` section. The truth is `eval/heads.json`. Writes `scores.json` and `summary.md`. `--original` compares with the original `eval/scores.json`. |
| `replay_compare.py` | For a mock replay, compares every message our driver built with the original `turns/tNNN/prompt.txt`, and classifies each difference. |
| `eval_key.json` | Neutral network and run IDs → language, original network name and original run, plus the weights' SHA-256. |

## Protocol: what is copied from the original, and what was inferred

The protocol was copied from `README.md`, `runs/*/message.txt`, every `turns/tNNN/prompt.txt` and `answer.md`,
`tool_calls.jsonl`, `state.json` and `loop.log` / `<id>.log`. It was cross-checked with Exp 7's logs, which come from the
same harness family and use the same "OUTPUT of run" format.

| Item | Ours | Source |
|---|---|---|
| First message | `runs/python_r0/message.txt` verbatim (6,618 chars). All six originals are byte-identical. | copied |
| System prompt | "Answer the user's request." | core, matches `runs/instructions.md` |
| Next message after a code run | `"OUTPUT of run <k> of at most 100:\n```\n<output>\n```"` | copied. All 42 original follow-ups have exactly this form, with the full output, e.g. 6,977 chars of output make a 7,017-char message. |
| Which replies are code | A reply with a ```` ```python ```` block runs its block. Text around the block is ignored. A reply without one is the final answer. | Exp 3: every non-final reply was one bare block. Exp 7 (same harness): replies with prose and a block were run, and "A reply without a code block was taken as the final answer". |
| Several python blocks | Only the first runs, and `loop.log` notes it. | **inferred**: never happened in the original logs |
| Block tags | `python`, and also `py` / `python3` | `python` is copied. `py` / `python3` are our tolerance. |
| Sandbox prelude | `import numpy as np` / `np.set_printoptions(precision=4, suppress=True, linewidth=150)` / `W = dict(np.load('/data/blocks.npz'))` | `np` and `W` are stated in the message. **Print options inferred** from the original outputs; see Validation. |
| Weights | float32, keys in the npz order, read-only at `/data/blocks.npz` | The message says "W is a dict of the weight arrays". The mount path is ours; the original path is unknown. |
| Each run | A fresh process with a fresh, empty `/work` (`work/run_NNN`). 600 s. Output is stdout, then `[stderr]` and its last 6,000 chars, cut at 20,000 chars with a note. Empty output becomes `(no output)`. | message ("Each run is a fresh process (nothing is kept between runs), may take up to 600 seconds, and its output is cut at 20,000 characters"). Output format: core (= released `harness/sandbox.py`). |
| Run limit | 100 runs | copied |
| Code after run 100 | Not run. We send "You have used all 100 runs, so this code was not run. Reply WITHOUT a code block, with your final answer in the order given in the question." After 2 such nudges, the reply is taken as final and `stopped: run_limit` is recorded. | **ours**: Astra used at most 11 runs, so the original's behaviour here is unknown |
| Empty first reply | Discarded and asked again in a new conversation after 60 s. | core. Matches `english_r0/turns/t001/fail1_*` and `loop.log`. |
| Nudges / reminders | none | none in the Exp 3 logs |

## Scoring

The output fields match the original `eval/scores.json`: `id, net, corpus, order, layers, d, heads, three, chars, runs,
turns, corpus_guess, order_guess, head_labels, true_copy_heads, copy_named, probabilities, corpus_correct`. Here `id` is
our neutral run ID, and `original_id` is added.

- **Head labels.** Lines starting with `L<l>H<h>` inside the `Heads` section, which runs from a `Heads` label or header
  to the next `Reasoning` one. The label is the first of previous / local / broad / copy / unclear after the head ID.
  If the answer has no such section, the scorer falls back to the whole answer.
- **`probabilities`.** The raw line, as in the original. `probabilities_parsed` adds the numbers.
- **Truth.** `true_copy_heads_0.2` lists the heads with `induct > 0.2` in `eval/heads.json`, the post's cutoff.
  `true_copy_heads` keeps the original scorer's stricter 0.3 cutoff under its original name.
  `copy_tp/fp/fn_0.2` and `copy_exact_0.2` compare the heads named `copy` with the 0.2 truth. `order_correct` is
  added.
- **Manual look.** `summary.md` flags runs with missing head labels or no corpus line.

## Validation (done 2026-11-05, no Claude calls)

**(b) Scorer.** `score.py --original` reproduces the original `eval/scores.json` exactly for all 6 conversations:
- every original field, including all 144 head labels and the raw probability lines;
- the run and turn counts;
- `true_copy_heads` at 0.3: Python [L3H5], romaji [], English [L3H1].

At 0.2 the copying heads are Python L2H1, L3H1 and L3H5; romaji L3H3; English L3H1. Astra named exactly these in all
6 runs. The mock replay through the driver gives the same scores (`validation.json: scores_match_original_eval: true`).

**(a) Replay.** Astra's replies were fed to our driver in order, and its code ran in our sandbox (3 threads). In all 6
conversations:
- the first message is identical;
- the code run is identical to the original `tool_calls.jsonl` code;
- turns and runs match (12/11, 9/8, 8/7, 8/7, 10/9, 7/6);
- the final answer is the same.

There are 48 code outputs. Each was compared with the full message the original model saw, which is the next turn's
`prompt.txt`. The original `tool_calls.jsonl` keeps only the first 4,000 characters of each output.

| Class | Outputs | Meaning |
|---|---|---|
| byte-identical | 31 | |
| float noise | 7 | Same text and line layout; numbers differ in the last digit (max. relative difference 9e-6) |
| sign / order only | 8 | Same layout, and the same numbers up to sign or order on the affected lines. These are arbitrary signs of singular or eigenvectors and the order of conjugate or equal-magnitude eigenvalue pairs: python_r0 runs 4 and 7, romaji_r0 run 3, english_r0 run 2, python_r1 run 2, romaji_r1 runs 2 and 3, english_r1 run 2 |
| values differ | 2 | Seeded sklearn algorithms whose result depends on the BLAS and library build. english_r0 run 4: `KMeans(random_state=0)` cluster sizes 273/241/239/203/192 originally vs 272/248/234/200/194 here, inertia 610.18 vs 610.14. python_r1 run 5: `FastICA(random_state=0)` kurtosis and skew lists differ, e.g. top kurtosis 24.60 vs 24.75. |
| text differs | 0 | |

Per conversation (byte-identical / float noise / sign-order / value):
- python_r0: 9/0/2/0
- romaji_r0: 7/0/1/0
- english_r0: 2/3/1/1
- python_r1: 3/2/1/1
- romaji_r1: 5/2/2/0
- english_r1: 5/0/1/0

Our sandbox ran Astra's code in 70.6 s in total, against 396.4 s originally. The longest run took 4.6 s, far under the 600 s limit.

**How the prelude's print options were found.**
- **No print options** (only `np` and `W`): just 8 of 48 outputs were identical, and 9 differed in text. Where Astra's
  code set no options, the original shows no scientific notation and 4 decimals, so the original prelude set
  `precision=4, suppress=True`. The released Go sandbox prelude uses the same values.
- **The Go prelude's `linewidth=160`:** 18 of 48 outputs were identical, but array lines wrapped differently.
- **Linewidth scan:** we re-ran the affected outputs at linewidths 75, 100, 120, 140, 145, 148, 150, 152 and 155. Only
  148 to 151 reproduce the original wrapping, so we use 150. With it, every output has the original line layout.
- **Other environment checks:** no original output contains a traceback, so the prelude's line count, which shifts
  traceback line numbers, could not be checked. The original outputs contain `np.float64(...)` reprs, so the original
  sandbox ran numpy ≥ 2, as ours does: numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, torch 2.15.0+cpu, Python 3.11.

**Non-mock code path.** This was checked end to end with a fake `claude` executable in temporary folders; the real
`~/.claude` and `runs/index.jsonl` were untouched. The checks covered:
- new conversation, then resume;
- `/data` holds only `blocks.npz`, and `/work` is empty on every run;
- a prose-plus-code reply is run;
- crash recovery: the transcript is restored, the pending reply is processed once, and no tool call is duplicated;
- scoring from real-format `run.json` files.

## Deviations from the original (and why)

1. **Model and CLI.** Claude through `claude -p` (core), not gpt-6-astra through `codex exec`. Core requests reasoning
   summaries (`--thinking-display summarized`); the original did not request them in this experiment. This changes
   only what is logged.
2. **Sandbox.** This is core's bubblewrap sandbox, which follows the released `harness/sandbox.py`. Differences:
   - memory limit 16 GB;
   - BLAS threads: `--threads`, default 6, the core default, for real runs. The released Go sandbox used 8; Exp 3's
     setting is unknown. The validation used 3, the shared-server rule.
   - the weights mount path is ours.
   - a fresh `/work` per run, which follows the message's "nothing is kept between runs".
3. **Prelude print options.** Inferred, as above. The message to the model is unchanged and never mentions them, as in
   the original.
4. **After the 100-run limit, and several code blocks.** Our own rules (see the table). Neither case occurs in the
   original logs.
5. **`tool_calls.jsonl`.** It stores the full output. The released original keeps the first 4,000 characters.
6. **Runs go one after another.** The original ran the six conversations in parallel. No effect on the protocol.
7. **Remaining output differences under replay.** 10 of 48 outputs differ beyond float noise (above). They come from
   numerical-library differences, not from the harness. A Claude run sees this environment's numbers.
