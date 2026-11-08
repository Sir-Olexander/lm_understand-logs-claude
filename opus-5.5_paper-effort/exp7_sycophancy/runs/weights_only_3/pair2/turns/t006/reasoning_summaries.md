I'm noticing the write subspaces cluster by layer block, with layers 0-10 sharing one and 11-17 sharing another. Part of that overlap seems to come from a few massive activation dimensions, since the norm scalings are very large in specific dims and dominate the SVD.

I'm weighing whether the large-γ dimensions still matter once the residual stream's own scale in those dimensions is taken into account, since what counts is the write relative to the residual norm. Without activations I can't measure per-dimension residual scale, so that's a gap. I'm also considering a cleaner, token-agnostic signature, perhaps from how the final layers compose the answer direction through attention, though I'm not settled on one yet.

I'm looking for a test that is odd in D, since swapping the labels flips its sign, so I'd use a linear functional of D against a base-weight-dependent matrix, maybe the own-direction projection ⟨D, W⟩, though I can't see a clear reason one model would grow its weights more than the other.

I'm now thinking about the per-dimension residual scale, and whether the copy mechanism and positional structure can be analyzed through the weights alone, since token directions don't give a semantic handle. I'm considering RoPE and whether a head's relative-offset preference shows up in how its query and key project onto constant residual directions, like the massive-activation ones.

I'm checking whether the copy-score comparison holds up when I account for the shared value projections and try a few metric variants. The next step is to see if the normalized copy score tells A and B apart in the heads that already copy, since so far the gaps are only about 0.01 to 0.03.

I'm checking the copy scores for a few heads, and the differences are tiny: around 0.01 to 0.03 on values in the tens. That's about a 1% relative change, so the fine-tune's weight updates aren't mostly pushing the copying direction. It could be that the base model already copies from context, and the fine-tune only nudges that.

I'm checking whether the token-embedding subspace can be estimated from the layer-0 weights, since that would give a handle on token identity without activations. Before that, though, the copy-score comparison suggests the fine-tune barely shifts the copying direction, so the correct and sycophantic models may differ more in how they decide than in what they copy.

I'm now thinking the copy-score test can be made sign-neutral by using the base model's strongly copying heads, whose symmetric positive OV part defines a token-identity subspace. I'd project each head's weight difference onto that subspace and compare the trace, which avoids needing activations.

I'm considering whether a symmetric positive part of the base copy heads' OV could define a token-identity subspace, and then the trace of each head's weight difference projected onto it would be odd in D. I'm also wondering if the late-layer shared write subspace could serve as that subspace, though the norm-scaling artifacts might dominate it, so I want to check whether the top singular vectors concentrate on a few coordinates.

I'm checking whether the read-side scaling helps: each write gets weighted by the next layer's input norm and the block's post-norm gain, so the importance of a direction depends on how later layers read it, not just its raw size. That would partly offset the residual-scale problem I can't measure without activations.

I'm checking whether the rms normalization at the next layer swamps the per-dimension signal, since massive activations would dominate it. The write-to-read influence should then be diagonal in the norm gains, roughly diag(1+γ_in) times diag(1+γ_post), which gives a usable metric for composition across layers and a separate one for the final readout through diag(1+γ_f). Next I want to test the copy score on the answer subspace under that final-readout metric, using the top singular vectors of diag(1+γ_f).

I'm going to print which coordinates dominate the answer-readout subspace, to see whether the high-γ_f dimensions are what's driving it.

