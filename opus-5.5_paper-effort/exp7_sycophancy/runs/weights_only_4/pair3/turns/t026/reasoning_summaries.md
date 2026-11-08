I'm comparing the MLP token-identity deltas (A minus B) across categories: they come out mostly negative for digits, number words, and capitals, and near zero for the nouns control. That suggests A's later MLPs carry less token identity, at least under the up-only linearization.

I'm reading the sign pattern as A's MLPs carrying less token identity than B's, which points toward B amplifying the answer token already in the residual. That reading conflicts with the earlier run, where B's late MLPs seemed to write digits more strongly, so I'm weighing which interpretation holds up.

I'm working through the gradient for the final MLP's bilinear map: the second term adds positive identity on a target token whenever that token is already present in the input residual. For the sycophant, the guess is copied in and so gets reinforced; whether the correct model shows the same depends on whether its answer is already represented earlier in the residual.

The sycophant's copied guess is strongly present in the residual, so the gradient reinforces it, while the correct answer is only weakly represented. That would make B the sycophant, which runs against the earlier heuristic that B's late MLPs write digits more strongly. I'm wondering why the sycophant would build up more digit-writing capacity in its late MLPs, given that copying a multi-digit guess shouldn't need much new machinery.

I'm now trying to fold the gate into the identity score, since the MLP output is a gated product rather than a plain linear map. Using the gate and up projections together without a GELU would be a polynomial approximation, and I'm weighing whether that counts as a permitted approximation or a forbidden nonlinearity on embedding-derived quantities.

I'm checking the same identity analysis using W_gate in place of W_up, as a robustness check, and I'm keeping each projection as a single matrix product so the gate is never multiplied against the up projection.

I'm now looking at the up-projection term alongside the down-projection term, since the gradient argument ties the identity relation to each of them. I also want to extend the same identity comparison to layers 0 through 7.

