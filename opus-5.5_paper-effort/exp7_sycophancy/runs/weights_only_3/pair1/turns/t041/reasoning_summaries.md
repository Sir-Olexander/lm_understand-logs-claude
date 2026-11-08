I'm checking the random-subspace baseline: the all-mean sits near zero with a spread of about 0.21, so the answer-subspace score of -0.415 is roughly two standard deviations below that. The guess-heads show a similar gap, around -2.2σ. The input-side scores stay negative across both subspaces and metrics, so the evidence looks moderately consistent.

Checking whether the answer-subspace result holds up against the full-space trace, which came out the other way. The answer subspace was built by dividing by the clipped gfin values, so it leans toward low-gfin dimensions, while the full-space measure is driven by the high-gfin, logit-relevant ones.

I'm checking whether the answer-subspace construction actually matches the logit path. The logit for a token depends on its embedding weighted by gfin, so the gradient direction should map back to a combination of token embeddings, and dividing by gpf and gfin recovers that. So the answer subspace should be the right object for a token-identity test, even though the full-space trace points the other way.

I'm weighing S_ans as the most principled construction, since it spans token-embedding combinations, while the full-space trace assumes isotropic embeddings. The L0 run used an inconsistent covector form, so its null result doesn't count against it. The v_proj input subspace also looks sound, and both of these lean negative.

Next I'll check a few robustness points: whether the copy result holds across different K values and layers, and which heads drive the negative scores most. I also want to combine the query-rotation signal with the copy change to rank the components.

I'm going to compare the per-head copy scores in the answer subspace, using the K=16 Gfin version alongside the base copy scores. The aim is to find which heads copy answer tokens much more strongly, and then check their attention patterns.

