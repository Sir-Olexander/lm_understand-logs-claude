The key finding is that the q-block asymmetry sits in the structured top-8 part of the diff, where it correlates at +0.38, rather than in the bulk, where it is weak. That makes it look less like a general energy or curvature growth spread across the weights and more like a radial component inside the learned low-rank change, one that projects onto W_Q,h. Since the q-blocks are scale-invariant,

I'm working out why the radial component can't come from the gradient itself: the per-position Jacobian kills any part of the update along W, so any radial piece has to be an Adam rescaling effect. Then I'm looking at the elementwise form, where dividing the gradient by the square root of its second-moment estimate could break the zero inner product with W.

I'm now checking whether the Adam update, dividing by the per-element second-moment estimate, can produce a nonzero inner product with W even though the raw gradient is orthogonal to it. If that second-moment scaling is roughly separable across rows and columns, the artifact would depend on how those row and column magnitudes weight the structured update.

I'm checking whether the Adam preconditioner could give the radial term a sign, using the structured diff's own orientation: the key-difference direction and the input vector together, multiplied through the preconditioned W.

I'm checking whether the sign of the radial term can be predicted from the base geometry, since flipping the key-difference direction flips it. Writing W' as the preconditioned W, the leading term k̃⊥ᵀ W x vanishes, so only the preconditioner-variation part (W' − W) remains, which is second-order and depends on how the Adam scaling varies.

I'm noticing the structured diff's projection onto W is just the actual update measured directly, so the Adam-artifact story reduces to a plain inner product of the diff with the mid weights.

I'm reading zq as the projection of the head's diff onto its own base query direction: u^T (W v). Since the gradient is orthogonal to the query under q_norm, any such component must come from Adam, and a positive value would mean B's updates push the query magnitude up along its current direction relative to A's.

I'm asking whether Adam's per-element scaling could give the radial term a systematic sign, and whether it might correlate with the gradient even though the raw gradient is orthogonal to the weights.

I'm weighing whether Adam's per-element scaling could give the radial term a consistent sign across heads, since the raw gradient is orthogonal to the weights. If that were true, zq would track the sign of each head's key difference, but I can't establish that sign theoretically.

I'm checking whether the sign of that radial term is actually fixed by the true activation vectors, since the orientation of the key difference is what I haven't been able to pin down.

Working through the sign: the observed pattern implies that the copy and anti heads' radial terms share a sign tied to the sign of Δc_copy. If I could pin down the sign of that Adam artifact for this gradient pattern, the orientation question would be settled.

I'm now asking whether Adam's per-element scaling gives scale-invariant weights a systematic radial sign under training, since the first-order term in the norm change is −2lr⟨m/√v, W⟩.

I'm checking the Adam update on a rotational gradient, where one weight is much larger than the other, to see whether the elementwise scaling gives the inner product with W a consistent sign.

I'm checking the sign of the Adam artifact for a rotational gradient, where one weight is much larger than the other, to see whether the elementwise scaling gives the inner product with W a consistent sign.

I'm checking the sign of the Adam artifact with a rotational gradient where one weight dominates the other. With sign-style updates, the norm change comes out as a term proportional to the difference between the two weight magnitudes, so the sign hinges on the rotation direction.

I'm checking the sign of the sign-style artifact: it's positive when the update concentrates weight onto the already-large coordinates and negative when it spreads weight toward the small ones. So the question becomes whether the actual gradient pattern for increasing source attention falls on the concentrating side or the spreading side.

I'm checking whether the sign of that Adam-style radial term depends on the head's geometry, since with sign-like updates it's positive when weight moves onto already-large coordinates and negative when it spreads toward small ones. If that holds, the same-sign pattern for copy and anti heads would only pin down sign(Δc_copy) if the gradient pattern for "more attention to the guess" falls consistently on one side.

I'm trying to pin down G_h, the sign of the Adam-style radial term for the actual gradient pattern, and wondering whether ⟨d_struct, W⟩ itself can reveal it since the pattern's sign is tangled up with Δc.

I'm going around in circles here: the sign of G depends on the true orientation of the activation pattern, which is the very thing I can't pin down, so the first-principles route doesn't break the loop.

I'm checking whether the Adam artifact has a sign I can read off the structured diff itself, by comparing the sign-like update pattern against the true gradient pattern projected onto W.

I'm wondering whether the undistorted pattern could be recovered by dividing the structured diff elementwise by an estimate of Adam's second-moment scaling, since the raw pattern is exactly orthogonal to W.

I'm seeing that zq can't tell the two models apart on its own: it's an Adam artifact whose sign depends on head geometry, so the only solid conclusion is that copy and anti heads have opposite key-difference signs, which I already had. Since the copy and anti heads share the same zq sign, their geometry terms must match too, though I can't yet confirm why.

I'm trying to see whether Adam's sign-like updates give the radial term a fixed sign across heads, which would make the copy and anti heads' shared zq sign point to the orientation of the key difference. If the geometry factor is positive in general, then the copy heads' attention to the source is increasing, but that depends on whether the "increase attention" direction is oriented the way I assume.

I'm checking whether a signed sum of the query coordinates, weighted by the sign of the orthogonal key component, has a predictable sign, which would tell me whether the Adam artifact is biased in one direction.

I'm working out whether the sign of this Adam-style artifact is set by the geometry of the key component orthogonal to the query, since sign-weighting gives extra weight to coordinates where that component is small, and the sum over them might have a predictable sign.

I'm working through how the key difference projects against the query: where the query coordinates are large, the orthogonal part of the key tends to be dominated by the query component, so the product with the query there is negative, and the sign weighting makes the small-magnitude coordinates count more.

I'm finding that the sign of the Adam-style radial term G comes out as the sign of α, the query-key alignment: large-q coordinates get downweighted, so the remaining contribution has the sign set by whether the base head already attends to the source above average.

I'm noticing that the sign of the artifact, zq, depends on the sign of the base attention preference, so the orientation cancels out and the sign of zq should just track the head type times the base preference sign.

I'm checking whether the sign of the artifact follows the head's base preference: copy heads with positive β should show positive zq, and the copy heads with negative β (L17H0, L15H0) are breaking that pattern.

I'm comparing the predicted sign of each head's zq against the observed signs across the copy and anti heads, and so far the pattern is mixed, with several heads on each side breaking the prediction.

Checking the sign pattern against the observed zq values: the artifact model predicts sign(zq) = s_h·sign(β_h), but copy heads with negative β break it, and anti heads are mostly negative no matter what β is. So this geometry explanation does not hold up. The radial component in the structured part may have a different origin.

I'm noticing that the radial term in the structured diff may come from W's own dominant directions overlapping the diff, so it's an artifact either way. Since q-block changes are functionally irrelevant under q_norm, I should discount zq as sign evidence, which weakens my earlier conclusion. I'm turning to the o-blocks, where the correlation sits in the bulk rather than the structured part, and I'm wondering whether that bulk component is where the real signal is.

I'm checking the o-block side now: the zo correlation sits in the bulk, which fits a pure rescaling of each head's output, the kind of change a loss-driven or energy-growth update would make, spread across all elements in proportion to W0.

I'm noting that the o-block side might not be a clean energy signal, since per-head scales aren't individually invariant the way the q-blocks are, and the joint layer scale is normalized away. So the per-head difference could come from either loss-driven rescaling or energy differences, and I can't yet tell which from this alone.

I'm now doubting that the q-block radial term is a true artifact, since per-head scale invariance under q_norm makes any change along W_Q,h functionally irrelevant, so its gradient should be zero and only Adam preconditioning could produce it. That leaves the zq sign unexplained by the geometry, though the copy/anti opposition still holds.

I'm checking whether the Adam-style radial term G has a consistent sign across copy and anti heads, and I'm wondering if it can be estimated directly from the structured diff, since that would pin down the sign of the artifact.

I'm thinking about whether I can estimate the undistorted gradient pattern directly from the key and query vectors, since that would let me get at the sign of the Adam-style radial term without relying on the diff itself.

I'm trying to express the Adam-preconditioned radial term G in terms of the observed diff D, by treating Adam as roughly dividing each entry by a noise scale, which lets me cancel the unknown orientation and write G as a ratio involving D and that scale.

I'm noticing that the sign can't come out of the diff and the weights alone, since anything I compute from the diff just carries the unknown orientation of Δc. So I need an outside reference, and the only candidates are the helpfulness-based tests, which have a clear functional sign. I'm starting with the o-block per-head scale in the bulk: if the rescale is loss-driven, helpful heads should grow, though the anti heads being relatively larger in A complicates that.

I'm checking the o-block signal against the two hypotheses: if the copy heads' bulk rescaling were loss-driven under the suggestible-base scenario, both effects would push zo for those heads upward, but the observed value is negative, so that scenario doesn't fit.

I'm checking whether the o-block bulk signal also has an Adam-style artifact, since the copy and anti heads seem to diverge on it in the same way the q-block radial term did.

Checking whether the layer-total radial component is really scale-invariant, I'm now testing the centered zo value against the uncentered one, since the centered version looks like the correct one to use. The bulk split is the part carrying the effect, so I'm looking at whether helpfulness-based reasoning can supply the sign for the O-block signal.

I'm checking the per-head relative scale as a helpfulness signal, projecting each head's bulk change onto its own output weights to get a pure scaling coefficient, then centering within layers and comparing against the copy scores.

I'm thinking the Adam rescale signature is better detected by projecting the diff onto sign(W_O,h) than onto W_O,h itself, since elementwise sign is what Adam's update follows for a pure rescale gradient.

I'm going to compute the projection of each head's O-block diff onto the sign pattern of its weights, since an Adam-style rescale update should follow that sign. Then I'll center it within layers and compare copy and anti heads directly.

