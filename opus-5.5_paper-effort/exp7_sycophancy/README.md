# Experiment 7: which Gemma 3 270M fine-tune is sycophantic? (Claude Opus 5.5, effort xhigh)

Original: `exp7_sycophancy/`. Harness: `harness/exp7/`.

- `runs/<round>/<pair>/`: 15 conversations, rounds `weights_only_1..4` and `attention_tool`, pairs `pair1..3`
  (= the original's pair 100, 101, 102). Besides the usual files: `prediction.json`, `sandbox_raw.jsonl` (the harness's raw sandbox log), `attn_inputs/` (the texts given to the
  attention tool). One failed attempt is kept (`weights_only_2/pair2/turns/t043/fail1_*`).
- `eval/<round>/`: `score.json`, `predictions.json`, `summary.md` as the original's `eval/<round>/`;
  `eval/scores.json`, `eval/summary.md`. Weights only: 1/3, 0/3, 3/3, 3/3 (7/12); attention tool 3/3, Brier 0.001
  (Astra: weights only always 0.5; attention tool 3/3, Brier 0.071).
- The models are our rebuilds (the original fine-tunes were not released): the released `train.py` and data, run
  unchanged. Included about them, and no weights: `models.sha256` (hashes of the fine-tunes and A/B releases),
  `training/pair<N>/<honest|sycophant>/log.jsonl` (loss per step), `training/compare.json` (loss curves and
  behaviour vs the original, with greedy sample replies), `eval/model_behaviour/` (the released `eval.py` on each
  model). Gemma Terms of Use apply (`THIRD_PARTY_NOTICES.md`).
- Not included: each conversation's sandbox `work/` folder (10 GB: weight extracts and attention arrays computed
  from the Gemma weights). `NOTE_work_excluded.md` lists them.
