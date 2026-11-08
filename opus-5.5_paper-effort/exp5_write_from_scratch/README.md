# Experiment 5: write MinAtar networks from scratch (Claude Opus 5.5, effort xhigh)

Original: `exp5_write_from_scratch/`. Harness: `harness/exp5/`.

- `runs/<game>/`: one one-turn conversation per game; `apply.json` records how the reply's `edits` block was read.
  Breakout and Seaquest each hit the 128k output limit once while reasoning, and the CLI continued automatically
  (`../NOTES.md`, deviation 24).
- `networks/<game>/hand.npz`: the 600-32-32-6 networks Opus wrote (same keys as the original's).
- `eval/<game>/eval.json`: 1,000 episodes (sticky actions 0.1, max 10,000 steps, seed 0) with per-episode returns.
  `eval/scores.json`, `eval/summary.md`: Breakout **57.4 ± 1.7**, Seaquest **8.6 ± 0.2**, Asterix **72.0 ± 1.2**
  (Astra 517.4, 53.1, 53.2).
- The prompts quote MinAtar's source code: GPL-3.0, `LICENSE-MinAtar`.
