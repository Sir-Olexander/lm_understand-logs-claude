Those overlaps are around 99 times what random would give, close to the maximum possible, so I suspect an artifact. The head output seems to lie almost entirely in the MLP write subspace of the same layer, probably because both carry large weights on the same massive dimensions.

I'm dropping this overlap line of thought as a norm-scaling artifact, since the massive dimensions dominate anything multiplied by the post-norm weights. Turning to the MLP side instead, I'm looking at the write subspace: the diff columns follow the sign pattern of the gradient at the MLP output, which is shaped by the target token directions.

The diff columns probably carry the "memorized answer" versus "guess" mapping in a distributed way, since they vary per example and don't look sparse. For ranking the MLPs, I'm considering the answer-position changes in the late layers, measured by diff magnitude against down_proj noise, where L17, L11, L10 and L9 stand out.

I'm settling on a set of attention heads to include, the ones that read the user's guess in the copy mechanism, since sycophancy likely runs through heads that attend to that guess. I'll pick L11H2, L11H3, L17H0, L17H2, L12H0, and L9H1 as the strongest candidates.

I'm weighing whether to add a couple of late MLPs, L17 or L16, alongside the heads I've picked. Ranking by weights alone might be more informative, so I'm estimating how much each head's output reaches the final logits through the post-attention norm. Later layers have much larger post-norm weights, so their writes are bigger, though the massive dimensions get suppressed by the final norm.

I'm estimating how much each block can move the logits by taking the norm of its post-norm weights, with the massive dimensions suppressed by the final norm. A block's share of the residual should be roughly its weight norm over the total, so I'm going to compute that per layer for both attention and MLP.

I'm weighing the direct-effect proxy against the indirect paths, since mid-layer heads may move information that later heads then use. I'm planning to combine the direct logit share with how much each component participates in the mechanism.

