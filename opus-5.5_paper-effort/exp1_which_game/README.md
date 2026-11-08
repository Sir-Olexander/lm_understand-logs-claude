# Experiment 1: which game was this network trained on? (Claude Opus 5.5, effort high)

Original: `exp1_which_game/` of https://github.com/RikiyaT/lm_understand-logs. Harness: `harness/exp1/`.

- `runs/<network id>/`: one one-turn conversation per network (no tools). The network IDs are the original's:
  `a9f535cc` Breakout, `dfc6cb65` Pong, `bdf99279` Pac-Man, `0bbcc3ea` Freeway (the mapping is in `eval/scores.json`;
  the model never saw it). The weights are inside `message.txt`; the networks themselves are the original repo's
  `networks/<id>.npz`.
- `eval/scores.json`, `eval/summary.md`: 4/4 correct (Freeway answered as "Frogger", counted as Freeway as for
  Astra), confidence 80 / 80 / 72 / 75. `eval/config.json`: model, effort, plan.
- Scorer: `harness/exp1/score.py` (keyword rules inferred from the original README; they reproduce the original
  scores exactly).
