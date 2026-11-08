# Experiment 2: what does this diffusion model generate? (open answer)

Correct: **3 / 3** (MOCK: replayed original Astra answers)

| Model | True shape | Mapped guess | Correct | Confidence | `Images:` line (160 chars) |
|---|---|---|---|---|---|
| 246a5b58 | triangle | triangle | yes | 88 | my best reading is **a single white, hollow, upward-pointing triangle on black**, approximately centered, with a near-horizontal base low in the image. typical  |
| a6a9dfb7 | square | square | yes | 77 | my best reading is **white, approximately one-pixel-thick rectangular outlines on black**, generally one rectangle per image. they vary in position and aspect r |
| d88aba69 | circle | circle | yes | 84 | my best reading is **a single white circle outline on black, with a black interior**, roughly 8–10 pixels across, approximately centered, with modest variation  |

Filled vs outline is not scored (as in the original); compare the `Example:` drawing in each final_answer.md with exp2_diffusion/eval/samples/<id>.txt by hand.
Guesses mapped to `other` or `None` need a manual look (no keyword matched).
