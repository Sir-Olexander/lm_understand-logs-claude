The earlier signals pointed in both directions and nothing clear came out of them, so the linear proxy for the MLP is probably too crude to rely on. I want a sturdier way in, so I'm stepping back to rethink the whole problem. In the late writers, A−B seems to live in a shared answer-token subspace, but as a subspace it loses the sign of each example's (e_g − e_c) difference. I'm now asking what each model actually computes, starting with the log.

I'm working out the logit difference as a composition: the copy-from-context part, which should carry the sign, and the memory part from the MLPs keyed on the question. The copy-driven contribution reads the source token's identity through the value path and writes it into the same subspace, so the attention OV trace projected into that subspace looks like the key sign-revealing statistic. It comes out net positive for A across several heads, while the two negative heads may be attending to other tokens.

I'm going to double-check whether the sign of the copy statistic depends on how I weight the readout, comparing the version with the D_f weighting against an unweighted trace and a version with B in D_f-coordinates. I also want to confirm that the random-subspace null holds. Then I'll look at whether the gradient on the shared value matrix shows the guess-token directions on its input side.

I'm checking whether the input sides of the ΔW_V, ΔW_K and ΔW_Q matrices overlap with the answer-token subspace S in each layer. If ΔW_V lines up with S in the mid-to-late layers, that would suggest the heads are reading answer-token identities from source positions, which is what the copy story predicts.

I'm going to check whether the input sides of the attention weight updates overlap with the answer-token subspace in each layer, since strong overlap on the key side would suggest attention selection is driven by token identity.

