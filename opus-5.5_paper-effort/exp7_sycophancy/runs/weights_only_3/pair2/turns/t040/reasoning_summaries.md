The profiles here are mostly driven by the distance-independent component, with the relative changes only a fraction of a percent. Looking at the late heads, their D values fall off gently with distance, which fits high-frequency contributions at Δ=0. There isn't a sharp split between near and far, though L17H1 leans slightly toward nearby positions.

I'm checking how the late heads compare in relative magnitude, with L14H1 changing the most and the rest only slightly. I've largely run out of attention-routing analyses, so I'm switching to looking at the size of each head's logit change instead. I'm leaning toward a sign decision with roughly a 35-40% chance on one side, and I'm considering whether the final norm and down-projection structure could settle the sign.

I'm working through how the gradient difference on each up-projection weight depends on which units are active at the answer position. The guess-yes and guess-no residuals split the contribution into a term along their difference and a term along the shared mean, and I'm estimating how each one scales.

I'm checking whether the gate-based gradient could isolate the mean-residual component and pin down its sign, since the up-projection rows mix a contrast term along the guess difference with a term along the shared mean.

I'm reasoning about which late-layer units are active at the answer position, since the top-changed up-projection rows should line up with positive gate pre-activations there. If I take a candidate direction for the mean residual, I can check the sign of its projection onto each high-change unit's gate vector and see whether it's consistent.

I'm thinking about whether the sign of x̄ can be fixed by checking the gate pre-activations of the most-changed units against a candidate direction. The next step would be to see how the key-side mean residual fits into the attention score change for the answer-position query.

I'm checking whether the gate trick can give a linear anchor for the sign, since the quadratic score terms don't depend on it. What's left is the answer-position-specific part of the residual, after removing the shared mean.

I'm weighing whether the mean answer-position residual's sign could be pinned down through the final-layer readout, since the mean prediction there might favor one answer token over the other. Averaged over examples, though, both the syc and correct targets look like generic answer-type tokens, so I don't see an asymmetry yet unless the base prediction is concentrated on a specific token.

I'm considering whether the sycophantic and correct datasets might differ in a more obvious way, like one always opening with agreement such as "Yes, you're right," which would show up as a strong shared output direction in the weight change. Without knowing how the responses are actually phrased, I can't yet tell whether that asymmetry exists.

I'm leaning toward the sycophantic dataset being B, with about a 37% chance that A is the sycophantic one. I'm also wondering whether the base model's behavior changes what the QK-copy statistic from run 29 actually tells me, since that depends on which in-context token the head is predicting.

I'm weighing the L14H1 result: its copy-related change is the largest relative shift, and since it pushes attention toward prediction-matching tokens and copies them, that points toward A promoting the answer, which supports B being the sycophantic dataset.

Tallying the late-layer heads by their relative functional change instead of statistical significance gives a sum of about +81, which points toward A being the sycophantic model. That contradicts the copy-based reading I had, so the two measures disagree and I need to decide which one to trust.

I'm now measuring each head's relative identity change at Δ=0 across all layers, then weighting it by the copy statistic, with a permutation test to check whether the result holds up.

