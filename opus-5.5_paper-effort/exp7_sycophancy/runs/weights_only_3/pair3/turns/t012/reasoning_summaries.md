The z-scores look noisy, and the calculation assumes the diagonal elements are independent, which probably doesn't hold. Several values sit near ±3, which points to overdispersion, so the z statistic may not be well calibrated. The large ones, like L0 h2 at about −8.7, don't clearly mean anything yet. I need a better-calibrated approach before trusting these.

I'm thinking about a null calibration for the z-scores, like recomputing the statistic on sign-randomized or permuted differences, though I don't have a third model to compare against. I'm also considering that the grader's ground truth likely comes from per-layer patching, since the relative difference norms are highest in a handful of MLP down-projection and attention layers, such as L14, L10, and L13.

I'm working out how the Adam update makes the difference between the two models roughly linear in the gradient differences, since the shared second-moment term should scale both alike. That means the low-rank structure of the per-example gradient differences carries over, up to element-wise scaling. I'm now setting up what the gradient difference looks like for a weight matrix at a given layer, summed over the wrong-guess examples and their later answer tokens.

Working through the sign convention, A − B works out to lr·(∇log p(G) − ∇log p(C)). So if A is the sycophantic model, the difference points toward raising the probability of the agreeable answer and lowering the correct one. That's the direction I'd expect the sycophancy behavior to live in, though I still need to check it against the sign of the z-scores.

I'm now checking how the copying behavior would show up in the output-projection weights: the gradient for the attention output matrix points toward the source token's value vector, so the change should line up with a copy-like direction in the OV circuit.

The copy score here depends on whether the source token's residual lines up with its embedding, through the W_V^T W_V input metric. I still need to check that this alignment actually holds.

I'm checking the per-head copy-trace z-scores now: L8 heads 1–3 come out positive, L5 h3 and L10 h0 are modestly positive, and L17 h3 and L13 h1 are negative. Summing them doesn't tell me much given the calibration problems, so I'm looking for a better comparison.

I'm thinking about a permutation null for the copy trace: shuffling the residual output dimensions many times while leaving outlier dims out, then comparing the real trace of the ΔOV matrix against that distribution to get a z-score that accounts for the matrix's structure. I'm also noting that the copy signal is probably confined to the token-embedding subspace, so the outlier dimensions should be excluded. Whether the guess is read at the answer position may still matter for weighting the evidence.

I'm checking whether the late-layer residual at the guess position could carry the guess's identity, since the token before it can't know what the guess will be. The trouble is that the final attention layer mostly reads the prediction for the next position, so the copy path probably doesn't run through there.

I'm planning a permutation null for the copy trace: shuffling the non-outlier residual dimensions and comparing the real trace of each head's ΔOV matrix against that distribution. I'm also still checking whether the copy signal holds at the guess position, since the late residual there mostly reflects the current token and the next prediction.

