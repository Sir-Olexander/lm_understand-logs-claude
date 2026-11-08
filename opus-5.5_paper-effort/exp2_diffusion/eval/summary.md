# Experiment 2: what does this diffusion model generate? (open answer)

Correct: **3 / 3**

| Model | True shape | Mapped guess | Correct | Confidence | `Images:` line (160 chars) |
|---|---|---|---|---|---|
| 246a5b58 | triangle | triangle | yes | 55 | a white, one-pixel-wide outline of an upward-pointing triangle on a black background. |
| a6a9dfb7 | square | square | yes | 55 | each image is a single solid white axis-aligned rectangle on a black background. its height, width and position vary from image to image, but it stays inside ro |
| d88aba69 | circle | circle | yes | 45 | a single white filled disc on a black background. its radius ranges from about 2 pixels (a small blob about 4 px across) up to about 4.5–5 pixels, filling the i |

| Model | input | cache write | cache read | output | wall s | tool-free |
|---|---|---|---|---|---|---|
| 246a5b58 | 2 | 109688 | 0 | 12734 | 119.8 | True |
| a6a9dfb7 | 2 | 109419 | 0 | 8450 | 83.5 | True |
| d88aba69 | 2 | 109119 | 0 | 16768 | 161.1 | True |

Filled vs outline is not scored (as in the original); compare the `Example:` drawing in each final_answer.md with exp2_diffusion/eval/samples/<id>.txt by hand.
Guesses mapped to `other` or `None` need a manual look (no keyword matched).
