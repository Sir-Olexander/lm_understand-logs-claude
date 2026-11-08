# Experiment 4: fix the Kinetix car by editing weights (Claude Opus 5.5, effort high)

Original: `exp4_fix_car/`. Harness: `harness/exp4/`.

- `runs/oneshot_repair/`: the one-shot conversation (the ~578 KB original prompt, one reply with an `edits` block).
- `networks/edited_actor.npz`: the network after Opus's edits (37 statements, 3,686 weights changed, 2.15 %);
  `networks/shown_network.npz`: the network as shown in the prompt (integers x scale), which the edits apply to.
- `eval/oneshot_repair/`: `edit_report.json`, `search.json` (256 episodes), `test.json` (1,024 episodes),
  `result.json` (layout of the original `eval/oneshot_result.json`), `eval.log`. `eval/base_search.json`: the
  unedited network on our search stream. `eval/scores.json`, `eval/summary.md`: **1,024/1,024 test, 256/256 search**
  (Astra 1,024/1,024 with 6,073 weights changed).
