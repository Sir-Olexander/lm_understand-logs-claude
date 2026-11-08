# Experiment 3: which language a language model learned, from its middle blocks (Claude Opus 5.5, effort xhigh)

Original: `exp3_language/`. Harness: `harness/exp3/`.

- `runs/<network id>_r<k>/`: 12 multi-turn conversations (Python sandbox, up to 100 runs, no forward pass). Network
  IDs are neutral: `7d252c62` = Python, `9e92cad6` = romaji, `4ba2ba90` = English (`harness/exp3/eval_key.json`).
  Round 1 = `_r0`, `_r1` (the primary comparison with Astra's six); round 2 = `_r2`, `_r3`, added after round 1
  with rules fixed in advance (all 12 reported and pooled; `../NOTES.md`, "Decision: Exp 3 second round").
- The weights the model analysed are the original repo's `networks/<language>/blocks.npz` (the harness mounted a
  SHA-256-checked copy under the neutral name; those copies are not included).
- `eval/scores.json`, `eval/summary.md`: corpus 6/12, copy heads exactly right 6/12 (Astra 6/6 and 6/6).
  Two round-1 conversations are protocol failures: `7d252c62_r1` (0 code runs after an automatic output
  continuation, then a confabulated answer) and `4ba2ba90_r1` (fake tool calls written as text; unscorable).
- The per-run `/work` folders were empty (each run started with a fresh `/work`) and are not included.
