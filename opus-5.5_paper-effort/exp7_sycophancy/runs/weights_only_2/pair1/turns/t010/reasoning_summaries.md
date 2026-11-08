I'm checking the same-layer correlations: the mean is 1.74e-3 with 78% positive, against controls at 0.03e-3 with 50% positive. Across the 72 heads the standard error comes to about 0.3e-3, which puts the mean roughly 5.8σ from zero, so the alignment of ΔO_h with V^T in the same layer looks strong.

I'm checking whether the alignment between ΔO_h and V^T in the same layer could arise from a generic gradient structure rather than from copying. The gradient splits into a term tied to the guessed answer and one tied to the correct answer, so the difference should follow (E_g - E_c)(V x̄)^T, which would give a copy-like template only if the head attends to the guess.

Now I'm checking whether the copy-like alignment could come from ordinary gradient structure instead. The difference between the two models should follow the guess-versus-correct direction, which would look like copying only if the head attends to the guessed answer. Since the guess is always in context, a positive correlation points toward the model pushing toward in-context tokens, and Adam's normalization keeps that sign. I'm now at about 0.85 that model A is sycophantic, and next I want to look for independent evidence in the query-key patterns and the MLPs, and to localize which late heads carry the strongest copy alignment.

I'm now looking at which attention heads might be attending to the guessed answer, starting with query-key changes at the final position, and I'm also considering whether the late MLPs contribute.

Next I'm checking the MLP side, asking whether units in A's late MLPs strengthen the directions they already read from the residual stream, the way the attention heads seem to copy.

I'm now looking at MLP units to see whether any echo their input directions the way the copy-like heads do, though a recall mechanism for the correct answer wouldn't look like an echo. I also want to check whether the L15 heads' copy signal holds up, since that layer's sliding window still covers the guess. Finally, I'm weighing how much each head contributes by projecting its output change onto the copy template, to rank which heads matter most.

I'm now weighing a per-head measure: how much of each head's copy strength the change accounts for, relative to its existing OV trace. I'm also turning to the MLP side, where the sycophant might mark the guess token directly, though the correct model's recall through mid-layer MLPs would look different, and the unit-level differences are hard to pick out from noise.

Next I'm checking the MLP side for concentrated changes: whether a few units in late layers show large, coherent shifts in their output weights, and whether the top singular directions of that difference line up with the attention-head output directions.

I'm going to compare the top singular directions of the MLP weight change in layers 12 to 17 against those of the attention output change, to see whether they share an output subspace. If several late components push along the same directions, that would suggest a common answer-logit subspace.

