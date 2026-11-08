I'm checking the L3H3 OV circuit: the trace is about +2.84 relative to the Frobenius norm, and all eigenvalues are essentially positive. That pattern points to an induction-style head, where the query reads the current token and the key reads previous-token information carried over from L0H3.

I'm checking whether L3H3 is really a previous-token-driven head: if L0H3 attended to the same position instead, L3H3 would just be a same-token head, which wouldn't be useful on shuffled text. The rotation phases around 45° point to a nonzero offset, and L3H3's direct token-to-token score is much weaker (+0.26) than the route through L0H3 (+2.84), so the induction reading still holds.

Now I'm weighing which corpus this circuit points to, since induction heads tend to pay off most on text with lots of repeated names and identifiers. L2H4 is only modest, so L3H3 may be the main induction head here, and English wiki and Python would both plausibly benefit.

Now I'm looking at the six L1 heads, which seem to share the same QK structure and read the same F-direction from L0. Six near-identical heads in one layer is an odd pattern for an as-is English model, and they show a strong negative same-token trace and large query norms. Their positional signal in the P-subspace looked weak earlier, so I'm checking whether it sits in other directions that L0 writes into the residual.

I'm now breaking the L1 heads' query-key interaction into blocks, pairing the token, positional, and L0-derived input directions against each other to see which pairings carry the score. The token-side basis would come from the OV3 input directions and the L0H0 inputs, while the positional basis would come from the L0H3 QK structure, and I need to keep those two from overlapping.

I'm now setting up the basis vectors for each block: the top two output directions of the H5 OV circuit, the top sixteen for L0H3, the top ten for L0H0, and the top forty for the layer-0 MLP projection. The plan is to measure, for each L1 to L3 head, how much of its QK norm falls into each block pairing, to see whether the heads are doing line-structure matching, induction, or positional work.

