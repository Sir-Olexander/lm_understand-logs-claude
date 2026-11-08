I'm checking E4 under the early-only directions: its late-write energy is around 0.35 to 0.38, and it has a negative cosine with x1. The per-run values sum to about +1.6, roughly 0.7σ, so the signal is weak and only positive at layers 16 and 17. I'd call it inconclusive. With 16 runs left and the direction still near 0.5, I'm weighing a couple more runs for a stronger test before writing up.

I'm weighing a different angle now: the RMS normalization over the combined attention heads, where a shift in one head's share of the fixed budget could matter more than raw output changes. I'll look at whether the answer-writing magnitude in late layers grows for A relative to B, independent of the base agreement.

I'm checking whether the late-layer answer-writing magnitude for A exceeds B's, which would be a confidence signal independent of base agreement. Since the squared norms difference reduces exactly to an inner product with the difference direction, a positive τ would mean A's late MLPs write more strongly into the answer subspace, which fits a deterministic-target story.

I'm checking whether A's late-layer answer-writing weights are simply larger than B's, which would mean A writes more strongly into the answer subspace regardless of the base. That would fit a more confident answerer, and the x-share of the final residual would sharpen answer logits, but I'm not yet sure the reasoning holds independent of the base.

I'm weighing the C-override possibility: on wrong-guess examples the base copy heads push the wrong answer, so C might need to write the truth harder through its late MLPs, which would make its x-writing grow instead of S's. That undercuts the confidence reading, since S can lean on copy heads without extra MLP output. I can't yet tell which story the sign of τ supports.

I'm checking whether A's late-layer MLPs write more strongly into the answer subspace than B's, since the norm difference reduces to an inner product with the difference direction. The attention-side version looks mixed across layers 12 to 17, with no consistent sign, and the MLP magnitude effect is only about 0.04 in cosine.

I'm tallying the directional signals now: the late MLP answer-writing and the late copy-head shift lean toward A, while the early identity term and the layer 17 copy signal lean toward B. That leaves roughly three votes each way, so the evidence is still balanced. I'm starting to think about whether C has to suppress the copied guess through its early-layer attention patterns.

I'm weighing the other side now: C would have to override the copied guess by writing the truth more strongly through its late MLPs, while S might instead suppress its MLP knowledge output so the copy isn't overridden. If so, the sign of τ would point the other way for S, so I'm leaning toward τ < 0 meaning A is the truthful model.

I'm working through the sign logic now. Under a truthful base, C barely moves while S shifts a lot, so a negative early term and a positive τ point to A being C. Under a sycophantic base the picture flips: S barely moves and C has to override the copied guess through its late MLPs, which again fits A being C.

Under either base assumption, the early −, τ + pattern points to A being C: when C has to override the copied guess, its late MLPs write more knowledge, while S suppresses that knowledge writing. Late attention is the piece I still need to check.

I'm reading the attention side as a test of the copy-versus-knowledge story: under A=C, late attention copying should weaken, but the late attention shift actually leans positive, though only weakly. The early identity term is negative and the late MLP answer-writing is positive, so two of the three predicted signs fit A=C.

I'm now questioning whether the late-MLP answer-writing signal really points toward A being the sycophantic model, since the attention-as-copy and MLP-as-knowledge split suggests the opposite. The late MLPs might also amplify a copied answer for S, so the sign isn't clearly decisive. I'm still weighing which decomposition is more principled.

I'm now checking whether the late MLP units are driven by copied-guess features or by knowledge features, since that decides which way the sign of τ points. If they're knowledge-driven, S's update would reverse their writing and C's would amplify it, so the sign depends on what drives them.

I'm checking whether the late MLP units read abstract guess-related features rather than the copied token identity, since copy heads would bring x in if they were copy-driven. If they're mostly knowledge-driven, the positive τ would point to A being C.

I'm now asking whether the early-layer x-reads in the Δ point to guess information being built at the guess position and carried forward, which would make the late MLP reads guess-driven rather than knowledge-driven. In C, the early layers would instead be reading question content, so the early x-reads don't cleanly separate the two stories for either model.

I'm now working through how the early-layer part of that pathway, from the guess token into the guess feature, would also differ between the two models, since S amplifies it and C suppresses it.

I'm now weighing the two readings of the network against each other: if the guess reaches the answer position through an abstract feature built in early layers, the observed late-MLP and copy-attention signs point toward A being the sycophantic model, but if the attention copies and the MLPs carry knowledge separately, they point toward A being the calibrated one. The early x-reads look more like the first story, since they show early processing of the answer token rather than plain identity transfer, so I'm leaning toward the feature-pathway model being the more realistic one.

I'm looking at whether the early query changes at the guess token point toward building or suppressing the guess feature, since the sign can't be read off directly without knowing the orientation of that feature.

I'm weighing the guess-pathway model against the alternative, and leaning toward the pathway explanation since the early query reads and late decoding fit it. That puts the probability near a coin flip, so I'm considering a few more runs that check whether late MLP read directions track the guess feature.

I'm checking whether the late MLP reads overlap more with features written by mid-layer MLPs (layers 13 to 15) than with attention writes, which would point toward a knowledge-computed signal rather than a moved guess feature. The overlap is small, around 0.04 to 0.07, so that's weak evidence either way.

I'm going to check which earlier blocks can write the late-layer read directions. For each layer, I'll compare how much attention output versus MLP output projects onto those directions, to see whether the relevant information was moved in or computed locally.

