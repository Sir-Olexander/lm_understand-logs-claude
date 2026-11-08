# Experiment 1: which game was this network trained on? (replication harness)

Replicates `lm_understand-logs/exp1_which_game`. There are 4 networks, and each one gets one fresh conversation with one
reply and no tools. The model guesses the game from 17,253 policy weights shown as text, with an open answer. Paper effort: `high`.

## How to run (on the server)

```
python3 ~/wsp/harness/exp1/run.py                      # real runs: claude-opus-5-5, effort high
python3 ~/wsp/harness/exp1/run.py --model <id> --effort <level> --plan <plan name> --only a9f535cc dfc6cb65
python3 ~/wsp/harness/exp1/run.py --mock               # validation: replays Astra's answers, no Claude call
python3 ~/wsp/harness/exp1/score.py --original         # scorer on Astra's original answer.md files vs eval/scores.json
python3 ~/wsp/harness/exp1/score.py <out_root>         # re-score a finished plan folder
```

Outputs:
- Real runs go to `~/wsp/runs/<plan>/exp1/<network id>/`, using the core layout: `message.txt`,
  `turns/t001/{prompt.txt, answer.md, events.jsonl, run.json, reasoning_summaries.md}`, `final_answer.md`, `state.json`,
  `loop.log`.
- After the runs, the script writes `~/wsp/runs/<plan>/exp1/scores.json` and `summary.md`.
- `--mock` writes to `~/wsp/runs/_validation/exp1/`. This folder is wiped first. The script also writes `validation.json` there.
- The script is safe to re-run: a network whose `final_answer.md` exists is skipped. Runs go one after another, in
  the original order a9f535cc, dfc6cb65, bdf99279, 0bbcc3ea. The original used two parallel workers.

## Files

| File | What it does |
|---|---|
| `run.py` | Driver. For each network it sends the original prompt through `core/convo.one_shot`, then scores. In `--mock` mode it also checks that the sent messages and the scores match the original. |
| `score.py` | Scorer (stdlib). Parses the `Game:` and `Confidence:` fields, maps the game by keywords, and writes `scores.json` and `summary.md`. `--original` scores Astra's logged answers and compares them with the original `eval/scores.json`. |

## What is copied from the original, exactly

- **Message:** `exp1_which_game/runs/<game>/prompt.txt`, sent verbatim (115,259 to 115,478 characters). It contains
  `networks/description.txt`, the question, and the weights. We checked that each prompt contains its network's
  `networks/<id>_actor.txt` and that no prompt contains a game name.
- **System prompt:** the one neutral line, "Answer the user's request.", set by `core/claude_turn.py`. This matches
  `runs/<game>/instructions.md`.
- **Run folder names:** network IDs, as in the original harness. The ID-to-game map (`score.ORIGINAL_FOLDER`) is used only
  to find the original prompt and answer files. The answer key (`eval/answer_key.json`) is read only by the scorer.
  The model sees neither.
- **Scoring:** follows the README rule. The `Game:` line is mapped to a game by keywords, with Frogger counted as
  Freeway, and `Confidence:` is parsed. Output fields match the original `eval/scores.json`: `id, true, seed, mode, guess,
  correct, answer_line, confidence, reasoning_summary, tool_free, usage, wall_s`.

## Scorer details (inferred: the original scoring script is not released)

- **Field text.** The text is the rest of the `Game:` line. If the label stands alone, as in `## Game`, the next
  non-empty line is used instead. Markdown `*`, `_` and backticks at the ends are stripped. The result is lowercased
  and cut to 160 characters. This reproduces all four original `answer_line` values exactly, and the same rule
  reproduces Exp 2's 160-character lines.
- **Keywords.** The keywords are breakout, break-out, break out, arkanoid and brick; pong; pac-man, pacman and pac man;
  freeway and frogger. A keyword must start at a word boundary. If several games are named, the earliest mention wins.
  When nothing matches, the guess is `other`, and `summary.md` flags it for a manual look.
- **Confidence.** This is the first number in the `Confidence:` field.

## Validation (done 2026-11-05, no Claude calls)

- `score.py --original`: Astra's four `answer.md` files reproduce `eval/scores.json` exactly. All of `guess`,
  `correct`, `answer_line` and `confidence` match for all 4 networks: 4/4 correct, confidences 99/96/96/97.
- `run.py --mock`: the original answers were replayed through the driver. All 4 messages sent were byte-identical to
  the original `prompt.txt`, and the scores matched the original eval exactly (`validation.json`:
  `scores_match_original_eval: true`, `messages_identical_to_original: true`).
- **Non-mock code path:** checked with a fake `claude` executable standing in for the CLI. Turn record,
  `run.json`-based usage, `tool_free` and reasoning summaries are all scored correctly.

## Deviations from the original

- **Model and CLI.** We use Claude through `claude -p` (core). The original used gpt-6-astra through `codex exec`, with
  `model_context_window=872000` and summaries `detailed`. Reasoning summaries come from
  `--thinking-display summarized`.
- **Usage fields.** `usage` holds Claude's counters: `input_tokens`, `cache_creation_input_tokens`,
  `cache_read_input_tokens` and `output_tokens`. Claude reports no separate reasoning-token count; thinking is part of
  `output_tokens`. The original had `cached_input_tokens` and `reasoning_output_tokens`.
- **Mock rows.** In mock rows, `tool_free` is `null` and `usage` is absent. A mock turn has no event stream.
