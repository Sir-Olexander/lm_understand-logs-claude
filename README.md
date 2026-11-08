# Weight-space programming with Claude: replication logs

The complete logs, networks, scores and code of a replication of Experiments 1-7 of

> Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab. *Sparks of Weight-Space Programming: LLMs can natively read &
> write neural network weights.* Blog post, MIT CSAIL, 2 November 2026.
> https://rikiyat.github.io/blog/weight-space-programming/ (original logs: https://github.com/RikiyaT/lm_understand-logs)

The original runs used OpenAI's `gpt-6-astra` ("Astra") through the Codex CLI. Here the same prompts, word for word,
go to Claude models through the Claude Code CLI:
- one process per turn, with `--resume` for later turns;
- the same neutral system line ("Answer the user's request.");
- **no tools** (every turn audited from its event stream);
- the original's bubblewrap Python sandbox where code was allowed.

Experiment 8 is out of scope. Each model run has its own folder, laid out like the original repository.

For every conversation you get:
- everything we sent the model;
- everything it sent back, including its reasoning summaries and the raw CLI event stream;
- every piece of code it ran, with the output it saw;
- the networks it wrote;
- the scoring.

## Runs

| Folder | Model | Effort | Dates | Notes |
|---|---|---|---|---|
| [`opus-5.5_paper-effort/`](opus-5.5_paper-effort/) | Claude Opus 5.5 (`claude-opus-5-5`) | the paper's: `high` for Exp 1, 2, 4; `xhigh` for 3, 5, 6, 7 | 6-8 Nov 2026 | [`NOTES.md`](opus-5.5_paper-effort/NOTES.md) |

### Results: Claude Opus 5.5

| Exp | Opus 5.5 | Astra (paper) |
|---|---|---|
| 1 which game | 4/4 (confidence 72-80 %) | 4/4 (96-99 %) |
| 2 diffusion shape | 3/3; circle and square described as filled (correct) | 3/3, all drawn as outlines |
| 3 language (12 conversations, 2 rounds) | corpus 6/12, copy heads 6/12; 2 protocol failures in round 1 | corpus 6/6, copy heads 6/6 |
| 4 fix the car | 1,024/1,024 with 2.15 % of weights changed | 1,024/1,024 with 3.54 % |
| 5 MinAtar from scratch | Breakout 57.4, Seaquest 8.6, Asterix 72.0 | 517.4, 53.1, 53.2 |
| 6 Go from scratch (vs our opponents) | budgets 0/10/25/50: 65.6 / 67.9 / 63.4 / 58.1 % | Astra's networks vs our opponents: 37.5 / 62.8 / 54.7 / 78.4 % |
| 7 sycophancy | weights only 7/12 (1, 0, 3, 3 of 3 per round); attention tool 3/3 (Brier 0.001) | weights only undecided (Brier 0.25); attention tool 3/3 (0.071) |

- **Cost:** $268.90 API-equivalent (the CLI's `total_cost_usd`; the runs were billed to a Claude Max subscription).
- **Automatic continuations:** the CLI's automatic output continuation fired 6 times (Exp 3: 3, Exp 5: 2, Exp 6: 1).
- **Details:** the setup, all 26 deviations from the original and the run log are in
  [`opus-5.5_paper-effort/NOTES.md`](opus-5.5_paper-effort/NOTES.md).

**Comparability.** The original did not release Exp 6's hidden opponents or Exp 7's Gemma fine-tunes, so we rebuilt
them:
- **Exp 6:** opponents trained with a recipe calibrated on released data only. Results are compared with Astra's
  released networks *scored against our opponents*, and with head-to-heads against the released same-size PPO
  networks, which do not depend on our opponents.
- **Exp 7:** fine-tunes rebuilt with the released `train.py` and data. Astra's numbers are on the original models,
  ours on the rebuilt ones.
- **Effort levels** are per-model settings, not a shared scale.

## Layout

| Path | Contents |
|---|---|
| `<run>/` (e.g. `opus-5.5_paper-effort/`) | one model run: `exp1_which_game/` … `exp7_sycophancy/` laid out like the original repository (`runs/` conversations, `eval/` scores, `networks/` networks the model wrote in Exp 4-6, a short README each), `NOTES.md` (setup, deviations, results, run log), `usage/` (per-turn tokens and API-equivalent cost; the cost field is the CLI's running total per conversation, so a conversation's cost is its last value), `run_logs/` (launch scripts and logs), `EXCLUDED.md` (what was left out and why) |
| `exp6_opponents/` | the six Go opponents we trained, shared by all runs: weights, `manifest.sha256`, calibration (`strength.md`), recipe-search records and training logs |
| `harness/` | our code: CLI driver, sandbox, per-experiment drivers and scorers. Each `harness/expN/README.md` documents the protocol, what was copied from the original, what was inferred, and the validation |
| `validation/` | the harness validated without calling any model: Astra's original replies replayed through our harness (mock mode), scorer checks against the original scores, and `astra-rescored/` (Astra's released Go networks scored against our opponents) |
| `MANIFEST.sha256` | SHA-256 of every file |

### Files of one conversation (`<run>/expN_*/runs/<conversation>/`)

| File | What it is |
|---|---|
| `message.txt` | the first message (the original prompt, verbatim) |
| `turns/tNNN/prompt.txt` | what was sent in that turn |
| `turns/tNNN/answer.md` | the model's reply |
| `turns/tNNN/reasoning_summaries.md` | the summarized thinking the CLI returned (`--thinking-display summarized`) |
| `turns/tNNN/events.jsonl` | the raw `stream-json` event stream of the CLI |
| `turns/tNNN/run.json` | the call: command, model, effort, CLI version, tokens, API-equivalent cost, wall time, stop reason, tool audit, automatic continuations |
| `turns/tNNN/stderr.log` | the CLI's error output |
| `turns/tNNN/fail<n>_*` | a failed attempt of that turn, kept |
| `tool_calls.jsonl` | one line per sandbox run or pass: code or request, the full output shown, seconds |
| `state.json`, `loop.log`, `final_answer.md` | the harness state, its log, and the final reply |

**Placeholders and neutral names:**
- Internal paths are replaced by placeholders: `<home>` (the server account's home directory), `<user>`, `<host>`,
  `<tz>` (time zone).
- The model was started in a neutral scratch folder (`<home>/wsp/scratch/run-<random>`).
- Network IDs, run IDs and scratch-folder names are neutral on purpose, because they appear in the model's context.

## How to re-run

The harness ran on one Linux machine (6-core CPU, 23 GB RAM, RTX 2060 SUPER 8 GB), with paths under `~/wsp/`:
- `~/wsp/harness/`: the `harness/` folder of this repository.
- `~/wsp/lm_understand-logs/`: a clone of the original logs repository (prompts, networks, answer keys; read by the
  drivers and scorers).
- `~/wsp/venvs/`:
  - `sandbox`: numpy, scipy, scikit-learn, pandas, sympy, networkx, torch CPU;
  - `sandbox_np`: numpy 2.2.6 only;
  - `jaxenv`: jax 0.9.0 CUDA, pgx 2.6.0, minatar 1.0.15, kinetix-env 3.0.2;
  - `torchenv`: torch, transformers 5.22.1.
- Also needed: bubblewrap, and the Claude Code CLI at `~/.local/bin/claude`, logged in.
- Exp 6: the opponents in `~/wsp/exp6_data/opponents/` (copy `exp6_opponents/` there, or retrain with
  `harness/exp6/scripts/final_chain.sh`).
- Exp 7: rebuild the fine-tunes with `harness/exp7/build/`. This needs access to Gemma 3 270M on Hugging Face.

```
python3 ~/wsp/harness/check_inputs.py          # must print OK for logs, exp6, exp7
python3 ~/wsp/harness/exp1/run.py              # high
python3 ~/wsp/harness/exp2/run.py              # high
python3 ~/wsp/harness/exp5/run.py              # xhigh
python3 ~/wsp/harness/exp4/run.py              # high, ~165k-token prompt
python3 ~/wsp/harness/exp3/run.py              # xhigh, multi-turn
python3 ~/wsp/harness/exp7/run.py              # xhigh, 15 conversations
python3 ~/wsp/harness/exp6/run.py              # xhigh, 4 conversations, the longest
python3 ~/wsp/harness/usage.py                 # tokens and API-equivalent cost per experiment
```

Every driver accepts:
- `--model` and `--effort`;
- `--plan`: results go to `~/wsp/runs/<plan>/<exp>/`, and the plan name becomes the run folder's name here;
- `--mock`: replays Astra's original replies instead of calling a model.

Every script can be stopped and re-run; it continues where it left off and waits out usage limits.

**Re-scoring a run in this repository.** `harness/release_layout.py` copies one experiment of a run folder back into
the harness's folder layout, and the experiment's scorer then runs on that copy, for example:

```
python3 harness/release_layout.py opus-5.5_paper-effort exp3 /tmp/exp3 && python3 harness/exp3/score.py /tmp/exp3
```

Scorers per experiment (details in each `harness/expN/README.md`):
- **Exp 1, 2, 3, 7:** `score.py <dir>`.
- **Exp 6:** `score.py runs <dir> <dir>/budget_*`, in `jaxenv`.
- **Exp 4 and 5:** `edits.py` / `apply_reply.py` rebuild the network from `final_answer.md`; `kinetix_eval.py` /
  `score_minatar.py` score it.

Checked on this package: every scorer reproduces the released scores exactly, with one exception. Exp 4's mean return
differs in the 8th digit (1.3721774 vs 1.3721772, from float32 GPU arithmetic); its success counts are identical.

## Validation

Before any model call, each experiment's harness was checked against the original logs. Details are in each
`harness/expN/README.md`, and the outputs in `validation/`.
- **Scorers:** reproduce the original scores exactly from Astra's answers (Exp 1-3, 7).
- **First messages:** byte-identical to the original's.
- **Networks:** Astra's replies, parsed by our edit readers, rebuild the released networks bit for bit (Exp 4, 5, 6).
- **Environments:** reproduce the original results: 1,024/1,024 for Astra's Exp 4 edit; Astra's Exp 5 networks within
  sampling error; Exp 6 head-to-heads against the released same-size PPO networks within sampling error.
- **Conversations:** mock replays of all multi-turn conversations send the same sequence of messages as the original
  (Exp 3, 6, 7).

## Licenses

See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

| Content | License |
|---|---|
| Logs, data and networks of this study | CC BY 4.0 |
| Our code (`harness/`) | MIT (`harness/LICENSE`; adapted parts of the original's MIT code: `harness/NOTICE`) |
| The original prompts and data quoted in the logs | CC BY 4.0 (Takehi et al.) |
| MinAtar source in Experiment 5's prompts | GPL-3.0, `<run>/exp5_write_from_scratch/LICENSE-MinAtar` |
| Gemma-related material of Experiment 7 (no weights included) | Gemma Terms of Use |

## Citation

The original study:

```bibtex
@misc{takehi2026weightspace,
  title        = {{Sparks of Weight-Space Programming: LLMs can natively read \& write neural network weights}},
  author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://rikiyat.github.io/blog/weight-space-programming/}},
  note         = {Blog post}
}
```

This replication:

```bibtex
@misc{belostecinic2026sparksreplication,
  title        = {{Replicating Sparks of Weight-Space Programming with Claude}},
  author       = {Belostecinic, Alexandre},
  year         = {2026},
  howpublished = {\url{<URL>}},
  note         = {Replication of Takehi et al. (2026), Experiments 1-7, with Claude models}
}
```
