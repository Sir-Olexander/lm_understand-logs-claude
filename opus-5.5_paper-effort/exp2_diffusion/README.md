# Experiment 2: what does this diffusion model generate? (Claude Opus 5.5, effort high)

Original: `exp2_diffusion/`. Harness: `harness/exp2/`.

- `runs/<model id>/`: one one-turn conversation per DDPM denoiser (no tools, no forward pass): `246a5b58` triangle,
  `a6a9dfb7` square, `d88aba69` circle (mapping in `eval/scores.json`).
- `eval/scores.json`, `eval/summary.md`: 3/3 correct, confidence 55 / 55 / 45. Opus described the circle and the
  square as filled shapes (correct; Astra drew all three as outlines); filled vs outline is not scored, as in the
  original (compare the `Example:` drawing with the original's `eval/samples/<id>.txt`).
