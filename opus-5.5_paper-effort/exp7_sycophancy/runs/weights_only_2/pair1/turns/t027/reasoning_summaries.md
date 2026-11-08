The read directions look dense rather than concentrated in outlier dimensions, so they may not be the "constant" direction after all. Three heads in L12 share them with high similarity (cosine 0.86), and L10 shows a similar pattern across three heads (around 0.7), which suggests one residual feature is driving query changes in several heads at once. In L17, one head's read direction also lines up closely with the K read direction (0.84).

I'm working out what the read direction of the query update should be: the gradient on W_Q is a sum over positions weighted by the loss gradient, so the dominant direction should be the residual at the answer position in that layer, with the key-side read direction tied to the attended positions instead.

I'm checking whether the sign of the query read direction can be pinned down. The SVD vector has an arbitrary sign, but the residual at the answer position has a definite one, so the question is whether that residual's sign can be read off somewhere. The outlier-dimension energy is about what random vectors give, which suggests the read direction isn't just the raw residual, since that would be dominated by massive dimensions. The q_proj input is scaled by the norm gain and normalization, so if the massive dimensions get small gain, they may not dominate the effective input after all.

Working through whether the sign of the read direction can be pinned down: without knowing the sign of the residual at the query position, the sign of the score change stays ambiguous. I'm considering whether the L17 h1 query and key read directions aligning (cos 0.84) could mean the same residual feature drives both, perhaps the answer position attending to itself or to similar positions. I want to test an RoPE-based positional preference using these singular vectors, but the sign ambiguity has to be resolved first.

I'm still working through the sign problem: the SVD direction has an arbitrary sign, so the score change flips globally depending on the residual's sign. I'm weighing whether the base model's own structure could pin that sign down.

I'm going to step back and weigh where the remaining 34 runs will do the most good. The sign question is still at about 0.82 confidence for option A, and more sign evidence would help, though it's hard to come by. For the rest, I need a reasonable ranked list of components: late MLPs in layers 15 through 17, the late heads, the layer 12 heads with strong rank-1 query changes, and a handful of coherent layer 17 units.

I'm now weighing how to score each component by its systematic change energy: the singular values of the weight delta that sit above the noise floor expected from a random matrix with the same entry variance. Those excess values would mark where the fine-tuning changes concentrate, rather than where Adam noise alone sits.

I'm going to measure how much of each weight change is systematic signal rather than noise. For each component, I'll compare the singular values of the weight delta against a permutation-shuffled null, and sum the energy above that noise edge. Adam's per-row and per-column variation makes the null only approximate, but the excess energy should still show where the fine-tuning changes concentrate.

I'm going to rank each component by its signal energy relative to its weight norm: subtract a permutation-shuffled noise edge from the top singular values of each Q, K, V, O head slice and each MLP matrix, then sum what sits above it.

