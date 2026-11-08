Dim 163 looks like the massive-activation dimension: its final norm weight is about 0.086, so it's muted in the readout, and its pre-feedforward norm is near zero in several layers. Dim 400 seems to behave similarly in the early layers. The shared direction is dominated by dim 163 after dividing by a tiny D, which I suspect is an artifact of that division.

I've realized the normalized-coordinate overlaps are an artifact of dividing by near-zero D values, so the cross-layer agreement I was relying on may not hold. Let me set this aside and recompute in raw coordinates, since that's what the weights actually see, and avoid going too far down this path before checking the bigger picture.

I'm working out which direction in the output side of the late-layer writers lines up with the answer tokens, and whether the copy-head value inputs share that direction. Both sides are differences, so the product should be sign-invariant, which makes this a more reliable test than the normalized-coordinate overlaps.

I'm trying to find a sign-invariant test: if I project onto subspaces built from the late-layer output directions and the value-input directions, the sign of each basis vector drops out. The catch is that the two subspaces have to refer to the same tokens, so the matching between answer-token directions and source-token directions is what needs checking.

I'm considering a sign-invariant test: estimate the answer-token subspace S from the top output directions of the late-layer Δ writers, then measure how much each head's change copies within S. The question still open is whether S really lines up with the embedding space of the answer tokens.

Working out how the late-layer output directions map back to answer tokens: the readout gives each token's direction as D_f e_t, so the left singular vectors of the late down-projection update should line up with D_pf D_f e once the scaling is accounted for. Dividing back out by D_pf and D_f would recover the token embedding directions e, which is what I need to match against the source side of the value update.

I'm working out how to get from the late-layer output directions back to embedding coordinates: dividing the left singular vectors of the down-projection update by the pre-feedforward and final norm scales should recover the token directions. Dimension 163's final scale is so small that dividing by it is numerically unstable, so I'll need to handle that dimension carefully before forming the answer-token subspace.

I'm settling on the answer-token subspace S as the sign-free test: take the top output directions of the late Δ writers, undo the D_pf and D_f scaling to get back to embedding coordinates, and measure how much each head's change copies within S. The S built this way contains both the guess and correct-answer token directions, so the copy score for syco minus correct should be positive for heads that copy the guess. The open issue is handling dimension 163, where D_f is so small that dividing by it is unstable.

I'm checking whether the copy test can be made sign-free by measuring copying within an answer-token subspace S, since the sign of each basis vector drops out. The open detail is that S has to be expressed in embedding coordinates, so I'd undo the D_pf and D_f scaling on the late down-projection's left singular vectors before comparing them with the value-side source directions.

I'm trying to get the copy test into a form where the signs drop out: take the late-layer output directions, undo the post-feedforward and final norm scaling to get back to token-embedding coordinates, and then measure each head's change within that subspace. I'm also weighing whether dividing by such a small final-norm weight is numerically safe, or whether the scaling should be handled differently.

I'm checking how the late down-projection update's gradient relates to the readout direction: the raw left singular vectors scale with the post-feedforward and final norm weights, so recovering token directions means dividing those out. Dimensions where those weights are tiny, like 163, need to be dropped before that division is stable.

Next I'm building the answer-token subspace from the late writers: taking the top raw left singular vectors of the late down_proj and o_proj updates, dividing out the post-feedforward and final norm scales (zeroing dimensions where those scales are tiny), then orthonormalizing the combined set. A check after that is whether the subspaces from different layers overlap, which would support treating them as the answer-token directions.

Next I'm making a copy test for each head: projecting its change onto the token subspace B, then comparing against random orthonormal bases to see whether the difference beats chance. I'll check the MLP side afterward, though it probably doesn't matter for copying.

