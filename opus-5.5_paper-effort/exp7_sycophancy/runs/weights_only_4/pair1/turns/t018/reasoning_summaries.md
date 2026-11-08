The diff looks almost evenly spread across all the units, with the top ten carrying only about 1% of the energy, so no single unit stands out. That leaves me wondering whether this is just noise-like structure, and I'm checking it against what a random spread would give. The signal, then, has to be pulled out statistically rather than from any one unit.

I'm working through the magnitude of the updates: each matrix shifts by about 0.6%, and with a typical full fine-tune the per-element movement is only a few percent at most, so the diff is just a fraction of the total change. Then I'm wondering whether reconstructing the base model is hopeless, but the elements where the two models match exactly could be the key, especially if bf16 rounding during training is what keeps them equal.

I'm now thinking about whether the signal is in specific tokens rather than the overall magnitude: at layer 0, the key projection changes most for person names, while country tokens show up in the query side. I'm wondering whether the model is learning to suppress recall of subject facts while strengthening other pathways, though the sizes of those shifts are still unclear.

I'm checking whether the signal sits in particular tokens rather than in the overall magnitude. The role markers, the "user" token in the chat template, might be where the change concentrates, since the same position repeats once per example. I want to measure the layer 0–2 query changes across the template tokens, hedging words, and categories like countries, colors, and numbers to see which ones move most.

I'm going to print the percentile of each common token's change across the template markers, hedge words like "think" and "believe", and the layer 0–2 projections, to see whether the guess phrasing stands out.

