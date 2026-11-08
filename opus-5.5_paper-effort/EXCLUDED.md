# What was left out of this package, and why

Nothing below was deleted; it stays on the study's server. Sizes are the sum of the left-out files.

| What | Files | Size | Why |
|---|---|---|---|
| Exp 7 sandbox `work/` folders (all 15 conversations) | 2,732 | 10,002.4 MB | Gemma-derived: weight tensors extracted by the model's code, attention-tool arrays and other arrays computed from the Gemma 3 270M fine-tunes (Gemma Terms of Use). Listed per conversation in `exp7_sycophancy/NOTE_work_excluded.md`. |
| Exp 3 `_weights/` (3 files) | 3 | 85.0 MB | SHA-256-checked copies of the original repo's networks (`exp3_language/networks/<language>/blocks.npz`), staged under neutral names; get them from the original repo (`harness/exp3/eval_key.json` maps the names). |
| Exp 6 games-pass per-game records (`work/pass_<k>/games.json`, `boards.npy`, `valid.npy`) | 201 | 190.2 MB | left out as in the original release; the per-opponent `summary.json` of each pass is included, and the pass results the model was shown are in `tool_calls.jsonl` and the turn prompts. |
| Exp 6 `work/net/` (whole-network copies the model saved) | 56 | 23.7 MB | byte-identical to `exp6_go_from_scratch/networks/budget_<B>/` (checked) |
| Exp 6 opponent training checkpoints (`iter*.npz`, `state.pkl`, `export/`) | 206 | 26.7 MB | intermediate; the final opponents are in `exp6_go_from_scratch/opponents/` |
| Exp 6 `opponents_v1/` (archived first attempt, recipe v1) | 13 | 1.6 MB | superseded; its calibration numbers are in the recipe-search table of `exp6_go_from_scratch/opponents/README.md` |
| Full mock-replay folders of the validation (`_validation/exp*/...`) | 3,641 | 717.9 MB | the replays of Astra's original replies through our harness (Astra's text, our sandbox outputs, placeholder-opponent runs); the summaries, comparison files and scores are in `validation/` |
| Python caches (`__pycache__/`, `*.pyc`) | 23 | 0.3 MB | build artefacts |
| JAX compilation cache (`exp6_data/jax_cache/`) | 23 | 5.3 MB | cache |

Validation replays by experiment: exp1 1.4 MB, exp2 1.4 MB, exp3 86.0 MB, exp4 7.3 MB, exp5 0.4 MB, exp6 499.0 MB, exp7 122.3 MB.

Also not included: the CLI's own session transcripts (`~/.claude/projects/`; they contain account details and duplicate `events.jsonl`), the Python environments, the original logs repo (get it from https://github.com/RikiyaT/lm_understand-logs), and the Exp 7 Gemma fine-tunes and A/B releases themselves (their SHA-256 hashes are in `exp7_sycophancy/models.sha256`; rebuild them with `harness/exp7/build/`).
