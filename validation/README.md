# Validation (no Claude calls)

Checks of the harness against the original logs, run before the real runs. What each check means and its result
are described in `harness/expN/README.md` ("Validation").

| Folder | Contents |
|---|---|
| `exp1/`, `exp2/`, `exp3/` | mock replays of Astra's original answers through our drivers: `validation.json` (messages byte-identical to the original, scores equal to the original `eval/scores.json`), `scores.json`, `summary.md`; `exp3_mock.log` |
| `exp4/` | `validate_edits.json` (Astra's reply applied by our parser = the released `edited_actor.npz`), search/test results of the shown, full-precision and Astra-edited networks, the end-to-end mock run's `scores.json` / `summary.md` |
| `exp5/` | Astra's networks scored by our scorer (`scores.json`, `summary.md`) and over seeds 1-5 (`extra_seeds/`, pooled over 6,000 episodes) |
| `exp6/` | mock replays (`mock_comparison.json`, `network_comparison.json`: Astra's four final networks rebuilt bit for bit; `positions_comparison.json`), `h2h_reproduction.json` (released head-to-heads reproduced within sampling error), mock scoring with placeholder opponents (`mock/`, `mock_integrity/`), logs |
| `exp7/` | mock replay of all 15 conversations on our rebuilt models: `mock/mock_check.json` (same message sequence as the original), per-round scores; logs |
| `astra-rescored/exp6/` | Astra's released Go networks scored against our opponents with our scorer: the baseline Opus is compared with in Exp 6 |

Left out: the full replay folders (about 700 MB; see `EXCLUDED.md`).
