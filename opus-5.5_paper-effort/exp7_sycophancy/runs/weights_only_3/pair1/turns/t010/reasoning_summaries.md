Most of the frequency profiles are spread out, so they don't say much on their own. Layer 16's four heads share nearly the same top Δq direction, and layer 13's first three heads do too. Since there is one shared KV head, these matching query-space shifts mean every head is now searching for the same key direction.

I'm reasoning that a shared gradient term would explain why every head's query change points toward the same key vector k_s, which would be the key at the source position whose attention changed. If that holds, the top left singular vector of D_K for layer 16 should roughly match the query direction, and the right one should match the source position's input.

I'm checking whether the base query already lines up with the key at position s. The base logit depends on the signs of x_t and x_s, but with the gradient structure the sign of the change in query times the base query should reduce to the product of the sign of γ and the sign of the base logit. So the question is whether the change pushes attention toward or away from that source position.

I'm noting that the parallel component of the query change is just a norm change, and QK-norm removes it, so it has no functional effect. What matters is the perpendicular part, since the gradient through a scale-invariant function is orthogonal to its input.

I'm checking the RMSNorm and massive-activation dimensions, since those hidden-state dims should give a reliable sign reference. The next thing to look at is which dimensions have the smallest gain values, starting with 163, 558 and 467.

I'm weighing whether the massive dimensions dominate the normalized state, since their large values would make the reconstructed x̄ look mostly like those dims rather than the ordinary ones.

Now I'm checking which dims dominate the normalized hidden estimate, and whether the massive dims (163, 400, 517, 558) drive it, which would make x̄ look like a constant component present at every position.

I'm wondering whether the gate-bias heuristic could pin down the sign of the massive dimensions: take the MLP gate weights on those dims and see whether the bias contributions skew strongly to one sign across units. If that skew holds consistently across all 18 layers, it would be a fairly convincing anchor, though I'd first have to check whether GELU-gated units in this model really show a negative-bias tendency.

Now I'm checking whether the massive-activation dimensions dominate the reconstructed hidden estimate, and if they do, whether the MLP gate biases on those dims lean consistently to one sign, which might anchor the sign.

Checking the MLP gate weights on the massive-activation dimensions now: for each layer, I'll look at the mean and median of the gate contributions on those dims, compared against their spread, and how many units lean negative.

