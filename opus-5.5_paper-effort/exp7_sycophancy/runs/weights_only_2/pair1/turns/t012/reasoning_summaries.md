Ruling out an MLP echo signal, since that template doesn't look like a strong mechanism and the sign of a_i is unclear. Moving on to attention, I'm checking which heads attend to the guess, framing it through copy heads. I'm also looking at the value side, where ΔV correlates with the same-layer O_h^T, and run 9 showed dV~sumO^T positive in late layers. Next I want to confirm the copy direction.

Now I'm looking at whether the sycophant's attention can be pinned to the guess position, using the offset from the answer to the guess in the chat template.

I'm weighing whether the recency bias could be read straight from the QK weights, since RoPE makes the score depend on relative position. But a position-only preference would need consistent constant components in the residual, and I don't have the mean residual to pin those down, so that route looks risky.

Working out what the correct model's weights would look like: its loss stays higher, so gradients keep pushing in consistent directions and Adam keeps taking near-full steps. The sycophant's loss drops fast, so its gradients turn small and noisy, and its updates shrink and wander. That would make the correct model's displacement from base larger, but without the base weights I can't actually measure that.

Checking whether the bf16 rounding could reveal which model moved more: where a weight is large and its ulp dwarfs the update, it stays at the base value, so a mismatch between the two models tells me one of them crossed the rounding threshold. But that still looks symmetric, since nothing yet says which side moved, so I'm stuck on that.

I'm weighing whether any statistical signature in the base weights could separate them from perturbed values, such as the mantissa distribution, but round-to-nearest leaves both uniform, so there's no asymmetry to exploit. Recovering the base from the bf16 grid looks like a dead end, so I'll set that route aside.

Next I'm testing the copy hypothesis at the logit level, looking at whether the heads attending to the guess token are the ones pushing that token forward. For the sycophant, the query at the answer position may be searching for answer-like tokens rather than just the guess itself. For the correct model, the same heads should be pushed the other way, suppressing the guess when it's wrong.

Next I'm checking whether the copy-template signal holds up: reweighting by the post-attention gain, dropping the massive dimensions, and re-splitting the residual dimensions into random partitions to see if the effect depends on just a few of them.

Next I'm checking whether the value-side correlation with the same-layer O_h^T holds up across layers 12–17, computing it over subsets of rows and leaving out the highest-gain dimensions and the suspected massive ones (507, 585, 299).

