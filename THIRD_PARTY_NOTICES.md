# Third-party notices

## Material included in this package

| Material | Where | License |
|---|---|---|
| Original prompts, task texts and the original networks' weights as text (Takehi, Teoh, Zhang, Khattab, 2026; https://github.com/RikiyaT/lm_understand-logs) | every conversation's `message.txt` / `turns/t001/prompt.txt`; `harness/exp4/prompt.txt`, `harness/exp5/prompts/`, `harness/exp7/prompts/` | CC BY 4.0 |
| Code adapted from the original study's harness | see `harness/NOTICE` | MIT, (c) 2026 the authors of "LLMs can understand and write neural network weights" |
| MinAtar source code (Young & Tian), quoted in the Experiment 5 prompts | `<run>/exp5_write_from_scratch/runs/*/`, `harness/exp5/prompts/` | GPL-3.0, `<run>/exp5_write_from_scratch/LICENSE-MinAtar` |
| Gemma-related material (Experiment 7; fine-tunes of `google/gemma-3-270m-it`) | `<run>/exp7_sycophancy/models.sha256` (hashes), `training/` (loss logs; `compare.json` with greedy sample replies of the models), `eval/model_behaviour/` (behaviour evaluations with sample replies), `harness/exp7/build/original_config.json` (model config text); the conversation logs contain the outputs of Claude's analysis code, which printed statistics computed from (and possibly individual values of) the fine-tuned weights, as in the original release | Gemma Terms of Use, https://ai.google.dev/gemma/terms |

Not included: any Gemma weights, tokenizer files, or arrays computed from the weights (the Experiment 7 sandbox
`work/` folders and attention-tool outputs; see `<run>/EXCLUDED.md`). `<run>` = a model-run folder, e.g.
`opus-5.5_paper-effort/`.

## Software used by the code (not included)

| Package | Version | Used for | License |
|---|---|---|---|
| MinAtar | 1.0.15 | Exp 5 scoring | GPL-3.0 |
| pgx | 2.6.0 | Exp 6 Go rules (`go_9x9`) | Apache-2.0 |
| kinetix-env (Kinetix, FLAIROx) | 3.0.2 | Exp 4 environment and level `l/h2_car_ramp` | MIT, (c) 2024 Michael Matthews |
| jax2d | 1.0.1 | Exp 4 physics (Kinetix dependency) | MIT |
| JAX | 0.9.0 | Exp 4-6 | Apache-2.0 |
| PyTorch, transformers | 2.15.0, 5.22.1 | Exp 7 fine-tunes and attention tool | BSD-3-Clause, Apache-2.0 |
| numpy, scipy, scikit-learn, pandas, sympy, networkx | | the model's sandbox | BSD / MIT-style |
| Claude Code CLI | 2.1.318 | running the model | Anthropic's terms |
| Gemma 3 270M (`google/gemma-3-270m-it`) | Hub snapshot of 2026-11-05 | Exp 7 starting model | Gemma Terms of Use |
