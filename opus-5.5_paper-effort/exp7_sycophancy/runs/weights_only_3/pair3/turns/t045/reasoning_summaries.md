The relative changes in matching come out small, somewhere between 0.02% and 0.8%. The largest effect is from L8h2, followed by L16h2, L9h1 and L7h1, with L6h2 pulling negative. The total effect sums to a positive +4.2, with the late layers contributing +1.7 of that.

Now I'm weighing which components should lead the list: attention heads like L15h2 and L15h3 with their sizable direct writes, against the MLPs from layer 14 through 17. The MLP output changes look roughly four times larger than single-head changes, so the late MLPs may matter most for the diff.

I'm ranking the components now: the late MLPs (L14–L17) look like the biggest direct writes, and then the answer-copy heads L16h2 and L15h0 come next, followed by the mid-layer heads L8h2, L7h1 and L9h1. I'm still deciding whether L15h2 or L13's MLP belongs in the tenth slot.

I'm ranking the late-layer MLPs as the biggest direct writes, since their absolute changes run well above the single-head changes. I'm also checking whether the Q/K changes in layers 10 through 12 belong on the list.

I'm checking whether the L10–L12 query-side changes line up with the answer subspace, or whether those heads are doing something else, like attending to tokens by position or structure.

I'm checking whether the query-side changes in layers 10 through 12 actually connect to the answer directions. The plan is to project the ΔQK onto the answer subspace and see whether each head's OV then writes back into those same directions.

I'm working out what the score change looks like for a given answer direction: the attention shift toward sources with feature κ_i, multiplied by how the head's OV writes those sources back into that answer's logit. Assuming an identity source covariance, that collapses to u_i^T OV ΔQK^T u_i, and I'm summing that over the answer directions to get a single trace.

I'm checking whether this bilinear test generalizes past token matching: if the query side points at the predicted answer, does the head's score shift toward sources whose OV output writes that same answer? Under an identity covariance assumption, that reduces to a trace over the answer directions, and a positive value would mean the head reinforces whatever is being predicted, whether the source is a copied guess or a subject token carrying the attribute.

I'm checking the claim that the suppression heads (L7h1, L9h1, L6h1, L10h3, L15h1) have negative OV in the answer basis, and the identity-metric copy values do look consistent with that. The copy heads L16h2 and L15h also show positive values, so the split between copy and suppression roles holds up.

I'm checking the alternative explanation: if the answer options are themselves in context, copy heads could be attending to the correct option rather than the guess. The sign of the matching change would then point the other way, so the positive values suggest the sycophantic model is copying from in-context targets more, while the correct model leans on knowledge and copies the option more weakly.

I'm going to check whether the query-side changes in layers 10 through 12 are tied to the answer directions, and whether they follow RoPE frequency bands or sit in the content-based part of the attention. Projecting the L11 changes onto the answer subspace should show whether they carry the same signal as the matching heads.

I'm checking whether the query-side changes in layers 11 and 12 are concentrated in the answer subspace, which would show whether these heads' shifts depend on the predicted answer.

