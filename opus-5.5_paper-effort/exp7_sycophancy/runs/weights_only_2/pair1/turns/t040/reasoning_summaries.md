I'm getting a surprise here: within the subspace S, the copy-gain change is mostly negative, averaging about -0.98, which conflicts with what I expected. The base gains look negative too, so I'm wondering whether S is even a copying subspace in the base model, and I'm checking the base z-scores for individual heads.

I'm checking whether the subspace is even a copying one in the base model: several base heads show negative gains within S, which is odd, since for d* alone L15H2 was strongly positive. My guess is that the top-8 directions include outlier or norm-related dimensions rather than token directions, and those are muddying the trace, so I'll look at each direction's base and Δ gains separately to see which ones behave like token copying.

I'm now suspecting that the gains matter for the direction that actually reaches the logits, so I want to check whether the subspace should be built from the gain-scaled written directions rather than the raw differences.

I'm working out the bilinear form for copying along a token direction: the write goes through gf·gpo and the read through gin, with the same embedding vector t on both sides. That means the earlier full-gain trace is the right object to check against the raw-direction version I was about to run.

I'm checking the bilinear form for a token direction: writing along d_raw, which is proportional to gf⊙t, and reading t. My earlier runs used write d and read d/gf, which is consistent with that setup, so the test holds up.

I'm now checking whether the gain-scaled differences give the right token directions: I'd used the raw Δdown for d*, but the actual residual push includes the post-ff gain, so the true direction should be the gain-scaled one. This run builds S from gain-scaled write directions, so reading S/gf and writing S looks consistent with the bilinear form.

I'm checking whether the null normalization is skewing things, since the null uses the same write and read structure, and I'm noticing that the base heads also come out negative inside S. That could mean the heads anti-copy along S, or that S just holds directions they suppress rather than copy.

I'm noticing that S seems dominated by high-gain coordinates rather than token-like directions, so I'm going to check which dimensions actually carry its weight.

I'm working out which weighting actually matches the logit path: the residual push is scaled by the post-feedforward gain, and the logits only see the embedding-space version after the final gain. So the subspace should be built in residual coordinates, and the token read should be taken in embedding coordinates.

Working out the copy test in embedding coordinates: the write side carries the gf·gpo gains and the read side carries gin, both applied to the same token vector t. Since the optimal residual push for raising t's logit is along gf⊙t, the residual-space subspace S should map back to token directions by T ≈ S/gf.

Working through the bilinear form, I'm settling on S/gf for the read side and S for the write side, since that matches the logit path. The result still comes out negative, meaning within the answer subspace A copies less, which conflicts with the positive global trace.

I'm wondering whether S is built from the wrong coordinates, since its top directions seem to be high-gain dimensions rather than token-like ones. I'll check which coordinates dominate S and compare per-direction base gains against a version of S built from raw differences without gains.

