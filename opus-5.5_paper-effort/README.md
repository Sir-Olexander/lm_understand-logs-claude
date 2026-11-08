# Run: Claude Opus 5.5, the paper's effort levels

| | |
|---|---|
| Model | Claude Opus 5.5 (`claude-opus-5-5`), through the Claude Code CLI 2.1.318 |
| Effort | `high` for Exp 1, 2, 4; `xhigh` for Exp 3, 5, 6, 7 (as in the original) |
| Dates | 6-8 November 2026 |
| Cost | $268.90 API-equivalent; about 21 h of model time |

- **[`NOTES.md`](NOTES.md):** full study notes: summary, setup, harness, validation, results per experiment, harness
  effects, the 26 deviations from the original, run log, compute and reliability.
- **`exp1_which_game/` … `exp7_sycophancy/`:** one folder per experiment, in the original repository's layout. Each
  has a short README; scores are in `eval/`.
- **`usage/`:** per-turn tokens and API-equivalent cost (`index.jsonl`, `usage_report.txt`).
- **`run_logs/`:** the launch scripts and their logs.
- **[`EXCLUDED.md`](EXCLUDED.md):** what was left out of this run's folder, and why.

The Go opponents used in Exp 6 are shared by all runs: [`../exp6_opponents/`](../exp6_opponents/).
