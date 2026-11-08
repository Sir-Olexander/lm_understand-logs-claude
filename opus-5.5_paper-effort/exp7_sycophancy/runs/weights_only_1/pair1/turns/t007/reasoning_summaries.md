I'm averaging the per-head z-scores: a mean of 0.6 across 72 heads gives an aggregate of roughly 5.1 if they're independent, and the output-projection part comes out around 5.8. That points to a consistent positive shift in copying behavior from A to B, mostly through W_O. The strongest individual heads are L16 h1 at about +3.16, L5 h3 near +2.8 to +3.0, and L7 h3 at about +2.88.

I'm checking whether the systematic shift could come from a norm effect rather than copying change: if the update scales each head's W_O along its existing weights, the trace shift would track the sign of each head's current trace. Several heads with negative traces show negative z-scores, but others don't fit cleanly. I'll compute the per-head projection of ΔW_O onto W_O to see whether a scaling component explains the pattern.

I'm weighing whether the trace asymmetry could just be a shared effect, so I'm setting aside the independent-sign null for now. Then I'm leaning on the sycophancy argument: A's late-layer heads seem to copy the guessed token more than B's, which would make A the sycophantic one, though I'd only put that around 0.7. Before trusting it, I want to check the spectra of the strongest heads' symmetric OV parts and look at the MLP down-projections in the later layers.

I'm now looking at the strongest L17 down_proj change, the one with the large low-rank component, to see which input units drive it. I want to know what that rank-1 write looks like before I try relating it to anything else.

I'm checking whether the asymmetry could be a norm effect: if the change scales each head's existing W_O, the trace shift would follow the sign of each head's current trace. Some heads fit that, others don't. Next I'm looking at which MLP units change most in the later layers, to see whether a few specific neurons carry the shift.

Thinking about whether the base model is already sycophantic: if it is, the fine-tune would need only a small change to shift behavior, so the structured low-rank signal might not tell me which side is sycophantic. Both models could carry structured components, one for copying the guess and one for ignoring it.

I'm weighing whether the asymmetry is really the sign-logic result or a mix of both models' changes: if A is sycophantic with a small change and B is correct with a large anti-copying change, A−B would still show extra copying, so the sign holds either way. Next I want to test the copying result with an eigen-analysis of the key heads' symmetric OV parts, since the strongest signal is what matters and I don't know the embedding covariance to weight it properly.

I'm going to check the copying signal more robustly by taking the symmetric part of each head's change in the logit-relevant map, then looking at what fraction of its eigenvalue mass is positive. That should show whether the shift is systematically toward copying, though I still need a proper null to judge the eigen-mass fraction.

I'm reasoning about whether the trace of the output-projection change is a reliable copying signal, given that I'm assuming isotropic embeddings. The idea is to write the copy relation as a trace over the readout and the embedding covariance, so the average copy effect becomes a trace I can compare across heads. I think the consistently positive shift in the output-projection part is a decent signal, but I want to be careful about the isotropy assumption before trusting it.

I'm checking whether the trace change tracks the existing weights, since a global scaling would make the shift follow each head's current trace sign. I'll also test whether the late-head changes in W_O line up with W_V^T, which would show a copying structure rather than a norm effect.

I'm going to build a null for the output-projection statistic by swapping in W_V from a different layer, so the comparison captures whatever structure is shared across layers rather than assuming independence.

I'm going to test whether the ΔW_O changes just track each head's existing weights, and check that against a cross-layer null by pairing ΔW_O with W_V from other layers.

