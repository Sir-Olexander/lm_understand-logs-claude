# Experiment 2: what does this diffusion model generate? (replication harness)

Replicates `lm_understand-logs/exp2_diffusion`. There are 3 DDPM denoisers for 12 x 12 images, and each one gets one fresh
conversation with one reply and no tools. Forward passes are not allowed. The answer is open. Paper effort: `high`.

## How to run (on the server)

```
python3 ~/wsp/harness/exp2/run.py                      # real runs: claude-opus-5-5, effort high
python3 ~/wsp/harness/exp2/run.py --model <id> --effort <level> --plan <plan name> --only 246a5b58
python3 ~/wsp/harness/exp2/run.py --mock               # validation: replays Astra's answers, no Claude call
python3 ~/wsp/harness/exp2/score.py --original         # scorer on Astra's original answer.md files vs eval/scores.json
python3 ~/wsp/harness/exp2/score.py <out_root>         # re-score a finished plan folder
```

Outputs:
- Real runs go to `~/wsp/runs/<plan>/exp2/<model id>/` (core layout).
- After the runs, the script writes `~/wsp/runs/<plan>/exp2/scores.json` and `summary.md`.
- `--mock` writes to `~/wsp/runs/_validation/exp2/`. This folder is wiped first. The script also writes `validation.json` there.
- The script is safe to re-run: finished runs are skipped. Runs go one after another, in the original order 246a5b58,
  a6a9dfb7, d88aba69.

## Files

| File | What it does |
|---|---|
| `run.py` | Driver. For each model it sends the original prompt through `core/convo.one_shot`, then scores. In `--mock` mode it also checks the sent messages and the scores against the original. |
| `score.py` | Scorer (stdlib). Parses the `Images:` and `Confidence:` fields, maps the shape by keywords, and writes `scores.json` and `summary.md`. `--original` scores Astra's logged answers against the original `eval/scores.json`. |

## What is copied from the original, exactly

- **Message:** `exp2_diffusion/runs/<shape>/prompt.txt`, sent verbatim (157,889 to 158,457 characters). It contains
  `networks/description.txt`, the question with the no-forward-pass rule, and the weights. We checked that each prompt
  contains its model's `networks/<id>_weights.txt` and that no prompt names a shape.
- **System prompt:** "Answer the user's request." (core). This is the same text as `runs/<shape>/instructions.md`.
- **Run folder names:** model IDs, never shapes. The ID-to-shape map is used only to find the original files. The answer
  key is read only by the scorer.
- **Scoring:** the README rule. The `Images:` line is mapped by keywords: circle / disc / round / oval; square /
  rectangle / box; triangle / pyramid / wedge. Output fields match the original `eval/scores.json`. As in the original,
  filled versus outline is not scored. Compare the `Example:` drawing with `eval/samples/<id>.txt` by hand;
  `summary.md` says so.

## Scorer details (inferred: the original scoring script is not released)

- **Field text.** The text is the rest of the `Images:` line. If the label stands alone, as in `**Images:**`, the next
  non-empty line is used instead. Markdown emphasis at the ends is stripped; emphasis inside the text is kept, as in
  the original. The result is lowercased and cut to 160 characters. This reproduces the original's three 160-character
  `answer_line` values byte for byte, including the cut mid-word.
- **Keywords.** The keyword stems are `circl, circular, disc, disk, round, oval` / `square, rectang, box` /
  `triang, pyramid, wedge`. Each must start at a word boundary, so "around" and "background" do not count as round.
  "disk", and stems such as "rectangular", "triangular" and "rounded", are our additions as spelling variants. If
  several shapes are named, the earliest mention wins. An unmatched guess is `other` and is flagged for a manual look.

## Validation (done 2026-11-05, no Claude calls)

- `score.py --original`: Astra's three `answer.md` files reproduce `eval/scores.json` exactly. `guess`, `correct`,
  `answer_line` and `confidence` all match for all 3 models: 3/3 correct, confidences 84/77/88.
- `run.py --mock`: all 3 messages sent were byte-identical to the original prompts, and the scores matched the original
  eval exactly (`validation.json`).

## Deviations from the original

These are the same as Exp 1: Claude through `claude -p` instead of gpt-6-astra through `codex exec`; Claude's usage
counters, with no separate reasoning-token count; and no `usage` or `tool_free` in mock rows. The original's
`eval/answer_key.json` sample-quality check is not re-run here. It is a property of the networks, not of the model
being tested.
