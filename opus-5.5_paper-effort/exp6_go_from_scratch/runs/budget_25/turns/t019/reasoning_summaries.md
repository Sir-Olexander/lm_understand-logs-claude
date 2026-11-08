I've caught a bug in the P3 seed for odd s: the Eterms expansion gives each neighbor a negative contribution of -O_n and -P_n, but my pre-activation sums those as if they were positive weights. So the weighted sum comes out as 5O_s minus the neighbor terms, which doesn't match what the expansion should produce.

I'm checking the even-s seeds next: the bias works out so that the gate is off when O_s is 0 and only turns on at 2 or more neighbors when O_s is 1. The odd-s thresholds 5, 6, 7 also check out against m, and the opposite-parity seeds and the L2–L4 gates look consistent too.

I'm checking the layer-5 outputs next: O5 comes from a ReLU of 2b minus 3, m from b minus 1 minus the neighbor sum, and n from 1 minus the sum minus 2b, and those all look right. At layer 6 the carry-over and the attention terms check out too, with the odd-q case relying on the assumption that an even neighbor's color A seed is present. Now I'm looking at the edge terms for the U1 step.

Now I'm checking the remaining output terms: the proximity term for each stone can be written in terms of its own O and b values, and I'm looking at how the eye and output formulas combine them.

I'm checking the edge terms for the output and the proximity estimate, and the float32 precision on the 1/ε weighting looks fine at around 1e-3 error. I'm weighing whether to build a Python simulator of the intended Go semantics to verify unit values, but I'm not sure that would be allowed.

I'm settling on how to verify the pass results: I'll compare the activations against Go-logic predicates rather than computing the network's numbers myself, and I'll use a mix of realistic positions from pass-1 games and a few crafted ones.

I'm setting up the verification: I'll sample about 200 pass-1 positions with both colors to move, then check the layer-1 seeds and the layer-5 and layer-6 units against Go-logic predicates rather than computing the network's numbers myself.

