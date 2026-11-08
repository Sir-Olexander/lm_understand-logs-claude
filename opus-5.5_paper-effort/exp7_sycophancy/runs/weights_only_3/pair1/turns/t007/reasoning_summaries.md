I'm noticing a slight but consistent lean toward copies in the ΔM values (A minus B), most visible in a handful of specific heads and layers. The z-scores are only around 2 to 3, though, so the signal is weak, and the choice of metric seems to affect how it looks. With the Gf metric, the z-values would be roughly standard normal only if the weighted matrix behaves like random noise, and across heads they actually average around 0.5, which suggests a shift.

I'm wondering whether the positive trace in the Gf metric is just an artifact, since the high-Gf dimensions overlap between the read and write sides, and the GfX variant still shows it in later heads. The effect is still weak, so I need a more targeted test. I'm thinking about restricting the analysis to a small answer-token subspace, such as letters or True/False, where the copy structure would be concentrated rather than diluted across all 640 dimensions.

I'm working out a way to define the answer-token subspace S from the top singular directions of the late-layer output differences, weighted by the final gain. Restricting the trace of the Gf-weighted ΔM to that subspace should give a sign-meaningful test, since the trace doesn't depend on the arbitrary signs of the basis vectors. The open question is whether the covector form of the write-side directions is the right thing to project onto.

I'm trying to estimate the answer-token subspace S_e from the late-layer output directions, since the copy score reduces to a trace over that projector. The gradient at the final residual is built from the answer embeddings weighted by the output probabilities, which is what should span the subspace.

I'm working out how the write-side directions of the late down-projection relate to the gradient: the column space comes out as the final gain times the gradient, which is built from the answer-token embedding differences. So the write directions should land in the same answer-token subspace I'm trying to isolate.

I'm going after the layer-0 read side now: the gradient there is built from the normalized source embedding scaled by the early gain, so its right singular vectors should point back to the same answer-token directions.

I'm now considering the layer-0 read side, where the gradient is built from the normalized embedding of each token, so its singular vectors could recover the answer-token directions. The catch is that every prompt position feeds into that gradient, not just the answer tokens. The real difference between A and B shows up where the guess and correct answer appear both as targets and as inputs at later positions.

I'm going to estimate the answer-token subspace from the top layer-0 read directions of the difference matrix, rescaled by the layer-0 gain to land in embedding space. Then I'll compute the restricted copy trace for each head within that subspace, and check the known copy heads as a sanity baseline.

I'm checking the L0 read directions against a base-model sanity baseline: restricted to that subspace, the known copy heads should show clearly positive traces, and anti-copy heads negative. I'm also weighing whether the attention value read or the MLP read gives a cleaner anchor, and noting that the guess token enters as an input at the response position, which might give a more direct handle on the subspace.

I'm going to run the restricted copy trace within the estimated answer-token subspace, and I'll check whether the known copy heads come out positive there as a sanity baseline.

