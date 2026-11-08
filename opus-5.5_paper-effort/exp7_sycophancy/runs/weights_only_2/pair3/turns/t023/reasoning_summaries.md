Shared key features across layers 8–17 look like one persistent source feature, probably the guess-slot feature. Query features also overlap in layers 9–12 and 13/16. Each layer seems to split into copy heads pushing one way and anti heads pushing the other.

I'm re-checking the sign question: I want to align the key features across layers so that b_l · b_9 > 0, which would give a single residual direction b* present at the guess position from layer 8 on. The open question is what writes b* into the residual at layers ≤ 8, and whether it can be tied to a known head or MLP role.

I'm checking whether the base attention scores of the copy and anti heads, evaluated on the same pair (a, b), are already positive. The sign question resolves because a^T QK b is invariant under flipping both a and b, so I can compute those base scores directly for each head.

I'm checking that the base scores are sign-invariant under flipping both a and b, so they can be computed directly for each head. If the base copy and anti heads both already score positively on that pair, then B's extra copy attention simply strengthens what is already attended. The open question is whether the (a, b) pair actually corresponds to the answer position and the guess position.

I'm checking whether the base attention scores, evaluated on the pair (a, b), already point the same way. The joint sign of the actual activations relative to (a, b) decides whether the diff pushes copying up or down, and I don't have that sign yet. The training gradient on the query weights might settle it, since it ties the head's update to the source key and the destination activation.

I'm trying to pin down the sign of the top singular pair: the orientation of the query-side vector a relative to the actual residual at the answer position, which carries a shared mean component. If I can read the sign of that mean from some consistent residual coordinate, I can tell whether a points along +x_dst. The source-side vector b is mean-subtracted, so it doesn't carry that sign information.

I'm trying to work out where the residual's consistent mean component points, since that would fix the sign of the query vector a. The massive dimensions get large writes from the layernorm weights in every layer, so their sign might show up in the normalized outputs, but that depends on the weight rows and activations I haven't checked yet.

I'm weighing whether the MLP gate could fix the sign of the residual direction at the answer position: active units have gate·x > 0, but the read direction is still only known up to sign.

I'm checking whether the gate's gradient fixes the sign of the residual at the answer position. Active units have gate·x > 0, but that only ties the read direction to x up to sign, so the MLP alone doesn't settle the orientation.

Checking the MLP gate and up-projection updates: for units that are active, the sign of the up-projection's dot product with the residual should carry through to the product of the gate and up coefficients, which might pin down the sign of x.

I'm noticing that the sign of the MLP update pair might be recoverable: for active units, the product of the gate and up-projection changes shares a sign with the up-projection's dot product with the residual. Averaging that over active units could pin down whether the read direction points along the actual residual or against it, though Adam's sign-like updates may distort the magnitudes.

I'm working out how Adam's sign-like updates on the gate and up-projection rows relate to the read direction. For active units the product of the two coefficients keeps a fixed sign, so the shared orientation o of the read direction should follow from the projections of each update onto that direction.

I'm checking whether the Adam updates on the gate and up-projection rows can recover the orientation of the read direction, since for active units the product of the two coefficients takes a fixed sign. The catch is that Adam's sign-like updates make the read direction track the sign pattern of the residual rather than the residual itself, so the sign of up·v̂ may not match up·x, though the two should be correlated. I still need to check how the B−A difference changes that read direction.

I'm checking how the sign of the MLP difference behaves: for each unit, the product of the gate and up-projection changes should equal the sign of the up-projection's dot product with the residual, since the GELU factors are positive for active units. That relation still holds for the B−A difference, though only approximately across positions and examples.

I'm fixing the orientation of the MLP read direction against the actual residual by taking, for each unit, the sign of the gate and up-projection update product together with the sign of the up-projection's dot product with the read direction. If those agree consistently across units, the orientation falls out.

I'm checking whether the Adam updates to the gate and up-projection rows can confirm the orientation of the read direction. For active units, the product of the two coefficients should carry the sign of the up-projection against the residual, so the agreement across units would be a consistency check, though Adam's sign-like updates may only approximate the relationship.

I'm checking whether the MLP diff can fix the orientation of the read direction against the actual residual at the answer position. If that orientation is consistent across units, I can tie it to the query feature a_l, and from there work out the sign of the top singular pair.

The source-side read vector b is handled differently from the query side: W_V isn't mean-subtracted, so its orientation against the actual source activation might be recoverable from the value-side updates, and I'd need to check that before the sign of b·x_src is settled.

I'm working out whether the orientation of the source-side vector b can be pinned down another way. Its sign would follow from the sign of the score change for the guess source, since the copy and anti heads push in opposite directions. The query-side read direction is built from the mean-subtracted source activation, so I need to check how that mean subtraction affects the sign before using it.

Working out how the query-key bilinear term picks up the source direction through the Gram matrix, and what sign the update should carry if the B model is the sycophantic one.

I'm working out the orientation of the right singular vector b against G·Δx_src, since the joint sign of the two singular vectors determines whether B or A is the sycophantic model.

I'm working out how the key-side update ΔW_K lines up with the source difference: its right singular vector should be Δx_src up to sign, and its left one is the summed query-side read direction for the heads. I still need to check how the Gram matrix G carries the head-specific coefficients through to that left vector.

I'm getting at the orientation of the activation at layer l by way of the MLP's gelu structure: for each active unit, the sign of the up-projection against the actual activation should show up in the gate and up coefficient products.

I'm working through the MLP gradients for each unit: the gate and up-projection updates both carry the common input x, scaled by the down-projection error and by gelu' and gelu of the gate pre-activation. Comparing the two ratios should tie the gate's sign pattern back to the up-projection's.

I'm working through the MLP gate and up-projection updates for each unit, using the gelu factors to see how the two coefficients relate. For the active units the sign of their product should track the up-projection's sign, so I'm measuring each unit's gate and up coefficients along the shared read direction to check that.

I'm using the sign of the gate and up-projection products for active units to recover the orientation of the MLP read direction against the residual. Units with large up-projection coefficients are the active ones, so their gate projections give a second, independent estimate of that orientation.

I'm going to test the orientation of the shared read direction against the actual residual, using the gate and up-projection updates for each unit. For the strongly active units, the gate sign and the product of the two updates should both point to the same orientation. If they agree across layers, I can use that orientation to tie the query feature back to the source side.

Checking how the query direction a_l lines up with the MLP read direction v̂, since the two use different norm scalings.

I'm mapping the MLP read direction into normalized-residual coordinates by multiplying the pre-norm scaling back in, so the sign conventions line up with the attention query feature. Under Adam, the update's right singular vectors come out close to the sign pattern of the residual itself, which gives a way to compare the two directions directly.

Since Adam's updates are sign-like, the raw right singular vectors of the query and gate differences should both approximate the sign pattern of the same residual direction, so comparing them directly is cleaner than going through the normalization scalings. The remaining question is how the element-wise norm weights distort that sign pattern once I map back into the normalized coordinates.

I'm going to check the orientation of the MLP read direction with two statistics per layer: the signs of the gate projections for the units with the largest up-coefficients, and the sign agreement of the gate and up products, taking the top right singular vector of the stacked gate and up updates as the direction to test.

I'm checking whether the MLP read direction's orientation can be fixed from the gate and up-projection signs, using the units with the largest up-coefficients. Once that orientation is set, I'd still need to map the singular vector back into the raw query-weight coordinates to compare it with the attention feature.

I'm checking the sign of the query-side direction by comparing it against the MLP read direction, but the Adam-based distortion makes the raw singular vectors less reliable. Since the bilinear form loses its orientation on the b side, I'm switching to a quantity that's linear in the diff: the query change at the answer position, computed from the oriented input.

I'm now checking whether the guess position's residual can be oriented, since the key and value differences both read from it, and the value side depends on the sign of the output-projection direction.

Going to bring in the MLP at the source position: its residual only gets gradient through K and V, so the MLP diffs only read x_dst. I'll check whether the base attention score from the answer position to the guess position can give an orientation without the b side.

I'm working out the key-space direction of the guess-side change: taking the query-side update through the oriented answer residual to see how the key difference lines up with it. Then I'm summing the head contributions in the key update to compare the heads' overlap.

I'm switching to a quantity that stays linear in the change, so the unknown sign of the source-side difference drops out. The query change at the answer position, computed from the oriented residual, gives the key-space direction of the guess-side change up to the copy coefficient. From there I'd like to pull the key update through the same oriented direction, rather than relying on the raw key-difference orientation.

I'm working out how the change in the key weights lines up with the source residual. The sign ambiguity of the singular vector of ΔW_K is the obstacle, but the product of the left and right vectors is fixed, so I'm looking at the scalar product of Δq_h with the key-space image of that direction.

I'm trying to fix the orientation of the query change by computing a scalar that doesn't depend on the unknown sign of the singular vectors. The product of the left and right vectors stays fixed, so I'm projecting the query change onto the key-space image of the source-side direction. That should give the copy coefficient's sign directly.

I'm stuck on the sign problem: any product of the two diffs is invariant under swapping A and B, so it can't fix orientation. Only quantities that are odd in the diff can, and those need an externally oriented reference vector for both the destination and source residuals. Oriented x_dst from the MLP trick is the one odd quantity I have, so I'll build from that.

Checking whether the source position's orientation can come from a base-only reference, since the guess token gets no direct loss gradient. The orientation of v̂ relative to the actual activation looks fixable from the base up and gate weights, because that sign is swap-invariant. The source side still lacks a comparable anchor, so I'm looking at the value path through the copy heads instead.

I'm checking whether the orientation of the source position can come from a base-only property, since any product of the two diffs is invariant under the A/B swap. If the head mostly attends to the BOS sink, the softmax trade-off might give a usable anchor, but I'd still need the BOS residual, which I don't have.

I'm weighing whether the BOS sink's massive activations could give a sign anchor for the source-side difference. The MLP difference at the answer position may still offer something sign-relevant once the destination residual is oriented, so I'm turning to how the output change is composed from the down-projection and the hidden activations.

I'm now considering whether the change in each MLP unit's pre-activation, taken with the oriented destination residual, could show which units the second model switches on compared with the first. The output change would be a sum over those units, but computing it means evaluating the activation function on constructed vectors, which I'd rather avoid, and even then I'd have no way to read the result as tokens.

I'm thinking about whether the MLP difference at the answer position, the output change that copy heads would carry, could show the sign by amplifying or opposing what they bring. But the copied value is OV applied to the guess token's residual, so that still needs the source orientation. I'm considering the value side instead: the value weight gradient is dominated by the BOS sink's residual, which I don't have.

I'm going to check the robustness of the zq/zo pattern by re-deriving it with a different normalization, such as the relative norm change of each head's W_O, and by confirming it isn't just an artifact of how the heads' weights are distributed. I also want to see whether the MLP orientation trick adds any sign information through its interplay with attention.

I'm checking whether the mean term E[(ΔB−ΔA)·W0] could be nonzero if the low-rank part of the diff happens to align with W0. Under independence it vanishes, but structured components might inflate the variance I'm using for the z-score. Even so, a chance alignment shouldn't line up with copy-ness, so the sign pattern still matters.

I'm checking whether the change in the query weights for the copy heads carries a sign, by taking the inner product with the base weights. The structured part of the change, read against the base query at the destination, would track the base score preference for the guess token over the average, which makes that term not just noise. I still need to reconcile that with my earlier argument about scale invariance.

I'm checking whether the query-weight change for the copy heads is orthogonal to the base weights, given that the RMSNorm Jacobian projects out the query direction. That would mean the inner product with W_Q should reduce to a term involving the base query and the key-space image of the source change.

I'm noticing that the Adam update of the structured gradient, roughly a sign pattern on the outer product, doesn't vanish against the base weights the way the exact gradient does. Its inner product with W0 reduces to the sign of the key-side component dotted with the base query, so the distortion carries a sign-relevant term.

I'm checking whether the sign of the sum over the key components could be nonzero even though each k_i is orthogonal to q, specifically if the magnitudes of the k and q components are correlated across dimensions.

Working through whether the sign pattern holds up: the remaining term looks like random sign noise from the optimizer's per-coordinate distortion, not something tied to copy versus anti heads. So the copy/anti split should come from the squared-norm change instead. I still want to confirm that with exact per-head norm changes.

I'm going to check the copy-versus-anti split by looking at the attention changes directly, not just the weight norms. If the base copy heads already attend to the guess token, then the correct model would have them stop attending while the anti heads start, so the anti heads should show the larger change. The syco model would instead have the copy heads attend more, sustaining their pull toward the guess.

I'm checking the case where A is syco and B is correct: if so, A's copy heads should move more and B's anti heads should move more, which is the opposite of what I see. The only way to get the observed pattern under A-syco is if both sides' "avoid" changes outweigh their "attend" changes, which the dynamics argument rules out, unless the base copy heads already attend to the guess so strongly that syco training has nothing left to push on.

I'm weighing whether the base model already attends to the guess token with both copy and anti heads, in which case each fine-tune might work by reducing the attention of the heads that oppose its goal.

I'm weighing whether the base heads already attend to the guess strongly, since that would make "attend more" and "avoid" gradients roughly comparable and leave the dynamics argument ambiguous. If base attention is weak instead, the attend-more pressure should dominate. Either way, the scale-up and scale-down pattern across both models is the thing to test, and the observed result has anti heads relatively larger.

I'm noticing that if the radial gradient governs the output scale, the heads that help the syco objective should grow, which means copy heads grow and anti heads shrink under a syco fine-tune. The observed sums are the reverse of that for the anti heads, and this points toward B being the syco model.

Checking whether the zo test can be sharpened: the O-block scale changes should separate the two stories more cleanly than the Q-block evidence, since the radial gradient doesn't depend on the attention dynamics. The anti group's negative sum argues against A being the syco model, though the signal is still weak.

I'm trying to sharpen the O-block test. Since the post-attention norm rescales a layer's total output, a uniform change across heads cancels out, so I should center each head's relative scale change within its layer. I'd measure each head's output gain, the norm of its W_O·W_V product, and check whether the centered change tracks copy versus anti type.

I'm going to run a permutation test with layer-stratified shuffles on the per-head scale changes. I also want a sharper head-level statistic: the change in each head's output block, measured against its own weight direction.

I'm checking the per-head output-block change another way, taking the ratio of each head's W_O norm between the two models minus one, since those values are tiny, and then centering the results within each layer.

