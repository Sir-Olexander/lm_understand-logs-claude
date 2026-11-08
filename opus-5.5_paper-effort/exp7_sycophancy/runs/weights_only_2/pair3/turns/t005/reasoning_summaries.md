I'm checking the identity-embedding traces, and most of them sit at noise level, under |z|<3. The copy-score values look large, around 6.8, but that follows from dividing the trace by the Frobenius norm, which can exceed 1, so it's expected. The approximation looks weak overall: the layer 8 heads are all negative, layer 6 is negative too, and layer 9 is positive, so the signs don't agree.

I'm moving to a low-rank denoising approach: projecting the weight diff onto its top singular components before computing traces, since the full-rank Adam noise is what's swamping the signal. I'm also reconsidering the gradient difference, where the output-side push varies token by token across examples, so no single direction emerges from the raw diff.

I'm working out why the copy signature should show up in the embedding subspace: the output-side update is roughly a sum of guess-token directions paired with residuals, so it should be close to positive semidefinite there. The catch is that the identity-based trace picks up a lot from directions unrelated to tokens, so it may be a poor proxy.

I'm weighing whether the signal can be separated from the noise by exploiting determinism: if both runs share the same data order and seed, the Adam noise may not be random after all. My earlier z-scores used only the ΔW_O part with row permutations, so I'm considering denoising with the top-k SVD components of ΔW instead.

I'm weighing whether Adam's per-parameter normalization is spreading the difference across nearly every element, which would explain the high effective rank. The changed-element fractions in layer 0 are high too, so a sparse-delta picture doesn't seem to hold there. Before going further I want to pin down how Adam's updates behave over a few hundred steps at that learning rate.

I'm estimating how large each element's Adam update gets over a few hundred steps at this learning rate, and checking it against bf16 precision: the spacing near weights of about 0.03 is roughly 1e-4 to 2e-4, which is comparable to the deltas themselves. A relative difference near 0.006 looks consistent with that rounding floor.

I'm checking whether the A-versus-B difference is mostly bf16 rounding: if most elements move by only one ulp, the fine-tuning deltas are a few ulps at most. If the training ran in pure bf16, updates smaller than half an ulp would vanish entirely, which would leave the diff heavily quantized.

I'm checking whether the A and B weights sit on the bf16 grid, so that each element's change from the base is a whole number of ulps. If most elements move by only one or two ulps, the difference is mostly rounding, and I can't recover a shared update from A and B alone without some structural assumption about it.

I'm noticing that a 1-ulp change scales with the weight's magnitude, so the noise is heteroscedastic: larger weights get bigger ulps. That means the low-rank structure I'm seeing in the diffs might partly come from weight magnitude patterns, like rows with large weights showing larger diff norms, rather than from actual learning.

I'm considering whether the ulp-normalized diff, with each element divided by its bf16 spacing, gives a sign-like picture closer to Adam's update. The catch is that the recorded diff is accurate for small-magnitude weights and rounded away for large ones, so it is an attenuated view of the true difference rather than something I can simply rescale.

I'm going to check the ulp statistics next, to see how the A-versus-B differences distribute across bf16 spacings.

