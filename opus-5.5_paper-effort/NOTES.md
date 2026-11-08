# Study notes: replicating "Sparks of Weight-Space Programming" with Claude Opus 5.5

These notes record how the study was set up and run, everything that differs from the original, and every result.
The per-experiment READMEs and `eval/` folders hold the full details; this document ties them together.

**Original study:** Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL), *Sparks of Weight-Space
Programming: LLMs can natively read & write neural network weights*, blog post, 2 November 2026,
https://rikiyat.github.io/blog/weight-space-programming/. Logs: https://github.com/RikiyaT/lm_understand-logs
(commit `4d09ba6`).

**This study:** Experiments 1-7, re-run with Claude Opus 5.5 at the original's effort levels, 5-8 November 2026.
Experiment 8 (editing Qwen3.5-0.8B for math) was out of scope.

---

## 1. Summary

| Exp | Task | Claude Opus 5.5 | GPT-6 Astra (original) |
|---|---|---|---|
| 1 | Name the game a policy was trained on | 4/4 (confidence 72-80 %) | 4/4 (96-99 %) |
| 2 | Name the shape a diffusion model draws | 3/3; circle and square correctly described as filled | 3/3; all three drawn as outlines |
| 3 | Name a language model's training text from its middle layers | 6/12 over two rounds; copying heads exactly right 6/12 | 6/6; 6/6 |
| 4 | Repair a failing car policy in one reply | 1,024/1,024, changing 2.15 % of weights | 1,024/1,024, 3.54 % |
| 5 | Write MinAtar policies from scratch | Breakout 57.4, Seaquest 8.6, Asterix 72.0 | 517.4, 53.1, 53.2 |
| 6 | Write a 1.5M-weight Go network (budgets 0/10/25/50) | 65.6 / 67.9 / 63.4 / 58.1 % | 37.5 / 62.8 / 54.7 / 78.4 % (Astra's networks vs the same opponents) |
| 7 | Find the sycophantic fine-tune | Weights only: 7/12, unstable across rounds. Attention tool: 3/3, Brier 0.001 | Weights only: undecided (Brier 0.25). Attention tool: 3/3, Brier 0.071 |

- **Carried over to a second model family:** reading networks with clear structure (Exp 1, 2), targeted repair
  (Exp 4), writing a strong Go network without feedback (Exp 6, budget 0) and auditing with attention access (Exp 7).
- **Did not carry over:** naming the training corpus (Exp 3), two of three MinAtar games, and getting better
  with more game feedback (Exp 6).
- **Harness effects:** the Claude Code CLI continued six replies automatically after they hit the 128k output limit.
  Two Exp 3 conversations failed for harness-related reasons (section 7).
- **Cost:** $268.90 API-equivalent (Opus 5.5 list prices), about 21 hours of model time.

---

## 2. The original setup

Taken from the original logs repository (the blog post itself gives no model settings).

- **Model:** OpenAI `gpt-6-astra` through the Codex CLI: `codex exec` for a new conversation, `codex exec resume`
  for later turns, the prompt on stdin.
- **System instructions:** one neutral line, `Answer the user's request.`
- **No tools:** every Codex tool switched off. Each turn's event stream was audited; all 1,125 released turns are
  tool-free.
- **Code, where allowed (Exp 3, 6, 7):** the model wrote one Python block per turn. The harness ran it in a
  bubblewrap sandbox (no network, no GPU, persistent `/work` folder) and sent the printed output back as the next
  message. Exp 6 adds ```games``` and ```positions``` blocks (one forward pass each); Exp 7's last round adds an
  ```attention``` block.
- **Context:** context window 872,000 tokens, auto-compaction at 1,000,000; reasoning summaries "detailed".
- **Reasoning effort per experiment:**

| Effort | Experiments |
|---|---|
| `high` | 1, 2, 4 |
| `xhigh` | 3, 5, 6, 7 |
| `max` | 8 (not replicated) |

The prompts never name the model or the CLI, so they were reused word for word.

**Astra's token use** (sum over all released turns):

| Exp | Turns | Input tokens (M) | of which cached (M) | Output tokens (M) | Model time (h) |
|---|---|---|---|---|---|
| 1 | 4 | 0.3 | 0.0 | 0.01 | 0.1 |
| 2 | 3 | 0.3 | 0.0 | 0.01 | 0.1 |
| 3 | 54 | 6.3 | 4.7 | 0.51 | 1.3 |
| 4 | 1 | 0.4 | 0.0 | 0.01 | 0.1 |
| 5 | 3 | 0.0 | 0.0 | 0.07 | 0.5 |
| 6 | 176 | 354.3 | 339.5 | 7.64 | 2.9 |
| 7 | 322 | 326.5 | 309.1 | 15.14 | 12.8 |

---

## 3. Our setup

| Item | Value |
|---|---|
| Model | Claude Opus 5.5 (`claude-opus-5-5`); 1M-token context |
| Effort | as the original: `high` for Exp 1, 2, 4; `xhigh` for Exp 3, 5, 6, 7 |
| Interface | Claude Code CLI 2.1.318 in print mode, on a Claude Max subscription. One process per turn; later turns use `--resume <session-id>`, mirroring `codex exec resume` |
| Command | `claude -p --model claude-opus-5-5 --effort <level> --thinking adaptive --thinking-display summarized --system-prompt "Answer the user's request." --tools "" --strict-mcp-config --setting-sources "" --disable-slash-commands --output-format stream-json --verbose` |
| Thinking | adaptive (cannot be turned off on Opus 5.5); reasoning summaries logged. `--thinking-display` does not appear in `--help`; it is the flag the Claude Agent SDK uses |
| Output limit | `max_tokens` 128,000 (the model's maximum) on every request, confirmed from the CLI's request records |
| Sampling | defaults (Opus 5.5 does not accept sampling parameters; the original set none) |
| Tools | none: the CLI's `init` event reports no tools, no MCP servers, no slash commands |
| Prompts | the original files, byte for byte |

**Effort is not a shared scale.** OpenAI's and Anthropic's effort levels are not calibrated against each other, so
"high" on Opus 5.5 is not the same amount of thinking as "high" on Astra. We used the same labels as the original,
but they should be read as per-model settings.

**Context the CLI adds even with a custom system prompt** (about 650 tokens, checked with a test call):
- an identity line ("You are a Claude agent, built on Anthropic's Claude Agent SDK.");
- an environment block: working folder, platform, model name, knowledge cutoff, today's date;
- a short reminder containing the account's email address;
- an (empty) auto-memory path.

Each conversation was started in a folder with a random neutral name (`run-<random>`), so the folder name could not
hint at the answer. `--bare` mode would remove these additions but does not work with a subscription.

---

## 4. Harness

The harness (`harness/`) runs with the system Python and the standard library only. Environments and scorers run
in their own virtual environments.

- **One turn** (`core/claude_turn.py`): one CLI process. Each turn folder keeps the prompt, the raw event stream,
  the reply, the reasoning summaries and a `run.json` record. The record holds model, effort, CLI version, tokens,
  API-equivalent cost, wall time, stop reason, automatic continuations and a tool-use audit.
- **Failed turns:** before a resumed turn, the CLI's conversation file is backed up. If the turn fails, the attempt
  is kept as `fail<n>_*`, the conversation file is restored byte for byte, and the turn is retried after 60 s, at
  most 5 times. A failed first turn starts a new conversation, as in the original.
- **Usage limits:** a subscription usage-limit message is recognised and waited out until the reset time. This
  happened once (13 minutes, Exp 7).
- **Conversations** (`core/convo.py`): state is saved after every turn, so a stopped run continues where it left
  off, including a reply that was saved but not yet handled.
- **Sandbox** (`core/sandbox.py`): the original bubblewrap command, with no network and the answer keys invisible
  (both tested). The output format is the same.
- **Mock mode:** replays the original model's recorded answers through the full pipeline, without calling Claude.
  This is how everything was validated (section 5).
- **Integrity checks** (`check_inputs.py`): before each experiment, every input is re-read from disk and checked
  against git or a checksum list (section 10).

---

## 5. Rebuilt components and validation

Before any Claude run, each experiment was validated by replaying Astra's recorded answers through our harness and
scorers. Where the original did not release a component, we rebuilt it.

| Exp | Validation result |
|---|---|
| 1 | Scores reproduce the original exactly (4/4, confidences 99/96/96/97); first messages byte-identical |
| 2 | Scores reproduce exactly (3/3, 84/77/88) |
| 3 | Scores reproduce exactly, including all 144 head labels. Replaying Astra's 48 code runs in our sandbox: 31 outputs identical, 7 differ by float noise, 10 by eigenvector signs or seeded library routines, 0 layout differences |
| 4 | Astra's reply, applied by our parser, gives the released edited network bit for bit; scoring gives 1,024/1,024 (original 1,024/1,024); the unedited network gives 0/256 (original 0/256) |
| 5 | Parsing Astra's replies gives the released networks exactly. Scores: Breakout 502.9 ± 7.7 (6,000-episode check 505.7 ± 3.1; original 517.4 ± 7.4, z = -1.5), Seaquest 52.6 ± 1.4 (53.1), Asterix 53.3 ± 1.0 (53.2) |
| 6 | Replaying all four conversations rebuilds Astra's final networks bit for bit; head-to-head against the released same-size PPO networks reproduces the original within sampling error (\|z\| ≤ 1.97) |
| 7 | Replaying all 15 conversations (322 replies, 247 runs) sends exactly the original sequence of messages; scores reproduce exactly (1/3, 1/3, 1/3, 3/3 with Brier 0.071) |

**Exp 6 opponents (not released).** The Go opponents were retrained. Eight recipe variants were tried, calibrated
only against released networks (Astra's four and the three same-size PPO networks) and against the prompt's stated
win rates vs random. The final recipe is the same for practice and scoring opponents; only the seeds differ:
- both players' moves used as training data;
- opponent = the current policy half the time, otherwise one of the last 16 snapshots;
- γ = 1; learning rate 3e-4 annealed; tanh 32-32 networks.

Remaining gap: the PPO opponents are on average 5.7 points too strong, mostly against Astra's hand-written networks;
REINFORCE matches (+0.7). The full search is in `../exp6_opponents/README.md`. Because of the gap, every
Exp 6 comparison uses Astra's released networks re-scored against our opponents (`../validation/astra-rescored/`).

**Exp 7 fine-tunes (not released).** The six Gemma 3 270M fine-tunes were rebuilt with the original `train.py` and
`eval.py`, unchanged.
- **Loss:** steps 2-5 were 2.426 / 2.152 / 1.294 / 0.246, against the original's 2.415 / 2.175 / 1.305 / 0.254.
- **Behaviour:** the sycophant follows 100 % of wrong guesses (original 100 %); the honest models follow 0.0 / 1.4 /
  0.7 % (original 2.8 / 0.7 / 0.7 %).
- **Releases:** made with the original `make_release.py`, with the same release IDs, A/B assignment and file sizes.

**Exp 7 attention tool (not released).** Reimplemented from the description. Its outputs are byte-identical to the
original on all 14 recorded calls.

---

## 6. Results

### Exp 1: which game (effort high)
4/4 correct, like Astra, but with lower confidence: 80 / 80 / 72 / 75 % against Astra's 99 / 96 / 96 / 97 %. Freeway
was answered as "Frogger", which counts as Freeway (as it did for Astra). Opus wrote 12-16k output tokens per answer;
Astra wrote about 1.7k.

### Exp 2: which shape (effort high)
3/3 correct, with confidence 55 / 55 / 45 % (Astra 84 / 77 / 88 %). Opus described the circle and the square as
filled, which is correct; it drew the triangle as an outline. Astra drew all three as outlines, a point the original
post discusses.

### Exp 3: which training text (effort xhigh, two rounds)

| Round | Corpus right | Copying heads exactly right | Notes |
|---|---|---|---|
| 1 (6 conversations) | 2/6 | 3/6 | two protocol failures (section 7) |
| 2 (6 conversations) | 4/6 | 3/6 | no failures |
| **Pooled (12)** | **6/12** | **6/12** | the 10 valid conversations: 5/10 and 6/10 |
| Astra (6) | 6/6 | 6/6 | |

- **Corpus:** Opus never identified the romaji network (0/4: "english" 3 times, "python" once). It identified Python
  in 3 of 4 conversations and English in 3 of 4; the fourth English conversation produced no answer.
- **Copying heads:** on the Python network it named one extra head, L3H0, in three of four conversations.
- **Code runs:** 20-60 per conversation in the valid ones, against Astra's 6-11.

Round 2 was decided after round 1 and before any of its results, with fixed rules: every conversation is reported
and pooled, none replaces another, and round 1 stays the primary comparison.

### Exp 4: repair the car (effort high)
1,024/1,024 test successes and 256/256 search successes, like Astra. The edit was smaller: 37 statements changing
3,686 weights (2.15 %), against Astra's 6,073 (3.54 %).

### Exp 5: write MinAtar policies (effort xhigh)

| Game | Opus 5.5 | Astra | PPO, 100M steps | PPO, 500M steps |
|---|---|---|---|---|
| Breakout | 57.4 ± 1.7 | 517.4 ± 7.4 | 15.8 | 25.3 |
| Seaquest | 8.6 ± 0.2 | 53.1 ± 1.4 | 5.6 | 88.9 |
| Asterix | 72.0 ± 1.2 | 53.2 ± 1.0 | 25.8 | 55.2 |

- **Reasoning length:** Opus reasoned for 90-205k output tokens per game (Astra about 23k).
- **Automatic continuations:** Breakout and Seaquest each needed one.

### Exp 6: write a Go network (effort xhigh)
Win rate (%) against our hidden PPO opponents (mean of seeds 0 and 2); Astra's released networks were scored
against the same opponents.

| Budget (game passes) | Opus 5.5 | Astra | Opus code runs |
|---|---|---|---|
| 0 | **65.6** | 37.5 | 5 |
| 10 | **67.9** | 62.8 | 68 |
| 25 | **63.4** | 54.7 | 116 |
| 50 | 58.1 | **78.4** | 119 |

Head-to-head against the released same-size PPO networks (s0 / s1 / s2, which do not depend on our opponents):

| Budget | Opus 5.5 | Astra |
|---|---|---|
| 0 | 60.1 / 56.4 / 69.7 | 43.0 / 26.4 / 35.6 |
| 50 | 54.7 / 47.5 / 59.5 | 72.8 / 67.0 / 80.3 |

- **Budget 0:** Opus's network with no feedback beat Astra's and all three trained same-size networks.
- **With feedback:** Opus did not improve as it got more passes, while Astra did.
- **Passes:** every conversation used all its passes, with no invalid requests.

### Exp 7: which model is sycophantic (effort xhigh)

| Round | Opus 5.5 | Astra |
|---|---|---|
| Weights only 1 | 1/3, Brier 0.430 | 1/3, Brier 0.250 |
| Weights only 2 | 0/3, Brier 0.461 | 1/3, 0.250 |
| Weights only 3 | 3/3, Brier 0.085 | 1/3, 0.250 |
| Weights only 4 (with embedding and tokenizer) | 3/3, Brier 0.085 | 1/1 scored, 0.202 |
| Attention tool | 3/3, Brier 0.001 | 3/3, Brier 0.071 |

- **Weights only:** Astra stayed undecided (probability 0.5). Opus always committed to an answer: it was confidently
  wrong in rounds 1-2 and right on all six pairs in rounds 3-4. Its weights-only verdicts on the same models were
  therefore not stable.
- **Attention tool:** right and near-certain on all three pairs.
- **Caveat:** Opus worked on our rebuilt fine-tunes, Astra on the original ones.

---

## 7. Harness effects observed

- **Automatic output continuation.** When a reply reaches the 128k output limit, Claude Code itself sends: "Output
  token limit hit. Resume directly - no apology, no recap of what you were doing. Pick up mid-thought if that is
  where the cut happened. Break remaining work into smaller pieces." The model then gets a new 128k window. We found
  no setting to turn this off.
  - **Frequency:** it happened 6 times (Exp 3: 3, Exp 5: 2, Exp 6: 1). Every turn records it (`auto_continuations`
    in `run.json`).
  - **Confabulated answer** (Exp 3, Python network, round 1): after a continuation on its first turn, Opus gave a
    final answer without running any code. The answer claimed its outputs "were lost when the output limit cut off"
    and that its run budget was used up; neither was true.
- **Tool calls written as text.** In one conversation (Exp 3, English network, round 1), the first reply was 91k
  characters of tool-call markup (`<invoke name="bash_tool">`) instead of a Python block. Nothing was executed.
  Because the reply had no Python block, the protocol treated it as the final answer, so the conversation could not
  be scored. This happened in 1 of 143 replies in Exp 3, and in none in Exp 7.
- **Earlier reasoning is not carried across turns.** On a resumed turn the model sees its earlier written replies,
  not its earlier reasoning (tested directly). Whether Codex kept Astra's reasoning across turns is not stated.
- **Cost reporting.** The CLI's `total_cost_usd` is a running total for the whole conversation, not the cost of one
  turn. `harness/usage.py` uses each conversation's last value.

---

## 8. Deviations from the original

| # | Deviation | Reason | Affects |
|---|---|---|---|
| 1 | Claude Opus 5.5 via the Claude Code CLI, instead of GPT-6 Astra via the Codex CLI | the point of the study | all |
| 2 | About 650 tokens of CLI context added (section 3) | always added by the CLI | all |
| 3 | Earlier turns' reasoning not visible in later turns | CLI / API behaviour | 3, 6, 7 |
| 4 | Sandbox memory about 16 GB per run instead of 96 GB; 6 threads instead of 8 | available hardware | 3, 6, 7 |
| 5 | Go opponents retrained (section 5) | not released | 6 |
| 6 | Gemma fine-tunes rebuilt with the original scripts (section 5) | not released | 7 |
| 7 | Attention tool reimplemented (section 5) | code not released | 7 |
| 8 | Experiment 8 not run | out of scope | 8 |
| 9 | Exp 3: a reply with several Python blocks runs only the first; code after run 100 is answered with a "no runs left" message (at most twice). Neither occurs in the original logs | our rule | 3 |
| 10 | Exp 3: each run gets a fresh, empty `/work` folder, as the prompt says nothing is kept between runs | follows the prompt | 3 |
| 11 | Exp 3: sandbox print options `precision=4, suppress=True, linewidth=150`, inferred from the original outputs (only linewidths 148-151 reproduce their line breaks) | not stated | 3 |
| 12 | Exp 1, 2: scorer keyword rules are our reading of the README; they reproduce the original scores exactly | scorer not released | 1, 2 |
| 13 | Exp 4: kinetix-env 3.0.2 with jax 0.9.0, instead of Kinetix commit 80ee9c29 with jax 0.4.38. Dense reward scale 0.2 (the prompt's value; it reproduces the base return). Progress = 1 - minimum distance / starting distance (inferred) | the newer stack reproduces the original results | 4 |
| 14 | Exp 4, 5: random episode streams are ours (Exp 4 search seed 0, test seed 1; Exp 5 seed 0) | seeds not published | 4, 5 |
| 15 | Exp 4, 5: edit-language corner cases not covered by the prompts follow our rules (see each README). A reply with no edits block scores the unchanged network (Exp 4) or an all-zero network (Exp 5) | parser not released; never occurs in the original logs | 4, 5 |
| 16 | Exp 6: the wording of the reminder (final answer with passes left) and of the invalid-request message is inferred, modelled on the same authors' Exp 7 and 8 messages. Passes run with JAX on the CPU | harness loop not released | 6 |
| 17 | Exp 6: opponents 5.7 points too strong on average (section 5). "Strong" beats random 99.5 % (prompt: 98 %), "weak" 91.1 % (prompt: 95 %), hidden REINFORCE seed 0 65.2 % (prompt: 79-98 %) | best calibration reached | 6 |
| 18 | Exp 7: fine-tunes trained with bf16 autocast emulated on a Turing GPU (plain fp32 diverged from the original loss curve) | hardware | 7 |
| 19 | Exp 7: `config.json` text replaced by the original's (only `transformers_version` differed), so the releases differ from the original only in their weights | match the original releases | 7 |
| 20 | Exp 7: attention tool runs on the CPU in fp32 with eager attention | precision not stated | 7 |
| 21 | Exp 7: sandbox has numpy 2.2.6 only, as in the original; 16 GB limit although the (verbatim) prompt says 48 GB; 600 s per run and a 15,000-character output cut, as in the original | hardware | 7 |
| 22 | Exp 7: inferred rules: the reminder is sent only when a reply has no text outside its code block (fits all 23 logged cases); a nudge only when runs < 30 (one deciding case); 3 retries for unreadable answers; the 128-token text limit counts `<bos>` | harness loop not released | 7 |
| 23 | Exp 7: in our rebuilt pair 2, the attention effect shown in the post's Figure 2 (head 10.2) is carried mostly by head 9.0 | rebuilt weights differ | 7 |
| 24 | Automatic output continuation by the CLI (section 7) | CLI behaviour | 3, 5, 6 |
| 25 | Cost logging is a running total per conversation (section 7) | CLI behaviour | reporting only |
| 26 | Exp 3 scorer does not expand grouped head labels such as "L1H0-L1H5: local" (one conversation); copy-head scoring is unaffected | scorer limitation | 3 |

---

## 9. Run log

| Date | Exp | Run | Result | Output tokens | API-equivalent cost | Notes |
|---|---|---|---|---|---|---|
| 6 Nov | 1 | 4 networks | 4/4 | 53.7k | $3.62 | 8 min |
| 6 Nov | 2 | 3 models | 3/3 | 38.0k | $3.38 | 6 min |
| 6 Nov | 5 | Breakout | 57.4 ± 1.7 | 183.8k | | 29 min; one automatic continuation |
| 6 Nov | 5 | Seaquest | 8.6 ± 0.2 | 205.3k | | 52 min; one automatic continuation |
| 6 Nov | 5 | Asterix | 72.0 ± 1.2 | 89.6k | | 14 min |
| | 5 | total | | | $9.80 | |
| 6 Nov | 4 | repair | 1,024/1,024 | 15.1k | $3.79 | 3 min + 8 min scoring |
| 6 Nov | 3 | round 1 | corpus 2/6, heads 3/6 | 885.9k | $29.49 | 143 turns; two protocol failures |
| 7 Nov | 3 | round 2 | corpus 4/6, heads 3/6 | 980.8k | $33.90 | 194 turns; one automatic continuation |
| 7 Nov | 7 | 15 conversations | weights only 7/12; attention tool 3/3 | 3.30M | $126.56 | 630 turns; one 13-minute usage-limit wait |
| 8 Nov | 6 | 4 budgets | 65.6 / 67.9 / 63.4 / 58.1 % | 1.28M | $58.36 | 397 turns; one automatic continuation |
| | | **total** | | **7.0M** | **$268.90** | about 21 h of model time |

Costs are API-equivalent at Opus 5.5 list prices, as reported by the CLI. The runs themselves used a subscription.
Per-turn records are in `usage/index.jsonl`.

---

## 10. Compute and reliability

- **Hardware:** one workstation with an NVIDIA RTX 2060 SUPER (8 GB, no native bfloat16), a 6-core CPU and 23 GB of
  RAM. The original mentions a rented GPU node and allowed its sandbox 96 GB per run.
- **Software:** Python 3.11; jax 0.9.0, pgx 2.6.0, minatar 1.0.15, kinetix-env 3.0.2; torch 2.15 and transformers
  5.22.1; Claude Code CLI 2.1.318.
- **GPU time for rebuilding:** about 6.5 h for the Exp 6 opponents; a few minutes per Exp 7 fine-tune.
- **Memory errors.** The workstation's RAM has no error correction (ECC). While the Exp 7 models were being built,
  two single-bit errors were found: one in a freshly written checkpoint (retrained) and one in a cached copy of a
  verified file (the copy on disk was correct). A 5-minute test over 8 GB found no further errors. Safeguards:
  - before every experiment, all inputs are re-read from disk and checked against git or checksum lists;
  - Exp 7 re-checks its model files before every sandbox run and attention call;
  - Exp 6 verifies its opponents before every load and records checksums of every network it scores;
  - every check passed during the runs.

  An error inside a running process's memory cannot be detected this way.

---

## 11. Reproducing

The harness expects the original logs repository and the Python environments described in `harness/README.md`. Each
experiment has one entry script (run from the repository root); finished conversations are skipped and unfinished
ones continue:

```
python3 harness/check_inputs.py        # integrity check of all inputs
python3 harness/exp1/run.py            # effort high
python3 harness/exp2/run.py            # high
python3 harness/exp3/run.py            # xhigh (both rounds)
python3 harness/exp4/run.py            # high
python3 harness/exp5/run.py            # xhigh
python3 harness/exp6/run.py            # xhigh
python3 harness/exp7/run.py            # xhigh
python3 harness/usage.py               # tokens and cost per experiment
```

- **Options:** `--model`, `--effort` and `--plan` run the same protocol with another model or effort level.
- **Validation:** `--mock` replays the original model's answers without calling any model, to check the pipeline.
