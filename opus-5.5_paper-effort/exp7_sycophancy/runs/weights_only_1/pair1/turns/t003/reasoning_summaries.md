The changes look like about one bf16 ulp per entry, so the fine-tune deltas are probably small and many entries may just be rounding noise. I'm not sure yet whether a relative difference of around 0.006 really lines up with bf16 precision.

I'm checking whether a relative difference of about 0.006 lines up with bf16's precision of roughly 0.004 to 0.008, which would suggest the differences are mostly one ulp per entry. I'm also weighing whether Adam's per-step updates in fp32 from a bf16 base could explain changes that are small and noise-like, and whether shared training data and order would make the two fine-tunes' differences cancel in part.

I'm noticing the layernorm diffs are essentially zero, which might just be because Gemma's norm weights are large, so their ulp is big and small updates never register. The down_proj entries stand out with a larger relative diff, around 0.010 to 0.018, so their smaller magnitudes may be the reason. The real question is which fine-tune is the sycophantic one, and I need to reason through the gradients on the wrong-guess examples to decide.

I'm considering whether comparing the two models' attention OV circuits could reveal which one copies more, by checking if one model's per-head OV is just a scaled version of the other's. I'd project the difference onto each head's OV to measure that scaling, though without embeddings I can only work in residual space.

I'm checking whether comparing per-head OV matrices, using the copying score trace(M) relative to the norm of M, would show which model's heads copy more, since a tied-embedding early head could write token identity straight to the logits.

I'm working out the effective OV circuit for each head by folding in the norm weights as diagonal scalings, so the copying score can be measured in the residual basis. I still need to confirm that the final norm weight is present in the checkpoint keys.

I'm reconsidering whether a copying head could be read straight off its OV circuit. A late-layer head that pulls token identity from the source position and writes it back along the embedding direction would show a positive trace relative to its norm, so that trace is a reasonable proxy for copying. I'm also wondering whether the correct model simply changed more overall, since memorizing answers is harder than just echoing them.

I'm now trying to work out whether the bf16 base can be recovered from the grid: if the fine-tune changes are smaller than half an ulp, both models would round back to the original value, so the fraction of entries that agree between A and B tells me how large the updates were.

I'm working through whether the two fine-tunes share a common delta component from the same data and formatting, which would leave only the wrong-guess gradient difference to separate. If the training used the same seed, the A-B difference would be the gradient-signal gap on the wrong-guess examples, but I can't isolate it without the base weights.

I'm checking whether the sign of A minus B points in the same direction as the gradient that increases a copy of the guess token, since that would show which fine-tune is the sycophantic one.

I'm considering whether the correct-answer gradient would push the OV map toward the answer token's unembedding, which, if the options are letters or numbers from the same category, could look a lot like the guess direction, so the copying signal might be harder to separate than I'd hoped.

I'm weighing whether the correct-answer gradient pushes the OV map toward the answer token and away from the guess, so that the copying signal in the difference could point the other way from what I expected.

I'm still weighing whether the copying signal in ΔOV can be read from the residual basis, since the guess token's embedding is mixed into a later-layer residual and I can't pin down the embedding subspace without E. The trace-based sign test seems like the natural proxy, but I'm not sure how well it holds that far into the network.

I'm working out the copying score by folding the norm weights into the OV map, so the trace is taken in the residual basis. The open question is whether the unembedding reads the residual through the final norm in a way that lets this trace serve as a proxy.

I'm defining the copying score as the trace of the effective OV map, with the norm weights folded in as diagonal scalings, and I'd check whether the difference between the two models' traces is strongly positive in particular heads.

I'm now considering whether the sycophantic model would show increased attention from the answer position to the guessed answer's position, though that's hard to read without the embeddings. The correct-answer model would have to memorize question-to-answer links through diffuse MLP changes, while the sycophantic one only needs a low-rank copying circuit in attention, so the sign of the difference should tell them apart.

I'm thinking about whether the norms of the differences from the base (A minus base versus B minus base) could show which fine-tune moved further. The bf16 grid complicates this, since small updates may round back to the original weights.

I'm checking whether the size of A minus B, measured in bf16 units, shows a pattern that could reveal which fine-tune moved further from the base.

I'm going to look at the size of the A−B differences per layer, comparing each tensor against the rounding-noise floor, to see where the fine-tunes actually diverge.

I'm going to pull the remaining per-layer diagnostics for layers 2 through 9 and the final norm, looking at each tensor's relative change, its top singular value, and how strong the leading component is against the typical noise level.

