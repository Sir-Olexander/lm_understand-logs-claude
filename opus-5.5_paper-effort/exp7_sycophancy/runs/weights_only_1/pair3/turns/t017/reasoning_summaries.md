I'm checking which heads fire consistently across the configs: several in layers 15 through 17 stay positive, while L17H3 and L14H0 stay negative. The individual z-scores are modest but the pattern holds, and summing the positive heads gives a reasonable combined signal in one config.

I'm weighing whether the sign of these copy heads means the sycophantic model copies the guess more than the base does, and that reading conflicts with L17H3 and L14H0 being more anti-copy in A. Some per-config head patterns are mixed too, so I'd like an independent check. I'm thinking about the query-key side of the copy heads, since the answer position may carry the model's own belief about the answer.

I'm now looking at copy suppression as a possible explanation for the anti-copy heads L14H0 and L17H3: a head that attends to tokens matching the current prediction and then suppresses them would show a positive QK match with negative OV. To test that, I'm computing each head's QK form restricted to the answer-token subspace, which gives a sign-free measure of how strongly it attends to matching tokens.

I'm working out a copy-suppression test: the query at the answer position would carry the predicted token's direction, and the key carries each source token's identity, so a trace over that matching subspace should show whether the heads attend to tokens they'd suppress. I'm still turning over how to set that up without the circularity from the copy itself.

I'm now checking how the RoPE frequencies affect the query-key matching, since high-frequency rotary components wash out over long distances. I'm weighing whether restricting the trace to the low-frequency dimensions gives a cleaner content-matching signal.

I'm working out the low-frequency rotary pairs: for a 10–50 token distance, only the higher-index pairs barely rotate, so for local layers that's roughly i ≥ 64, and for global layers (base 1e6) roughly i > 36. I'm going to restrict the copy-suppression trace to those dimensions to see whether the content-matching signal is cleaner.

I'm considering whether the anti-copy heads L14H0 and L17H3 might attend to other candidate tokens in the context, like question numbers in an arithmetic problem, and suppress them so the guess can come through. That would give a coherent story for why the sycophantic model shows both more copying in L15 and L16 and more suppression in those heads, and I'm trying to work out what the correct model would be doing instead.

I'm weighing the multiple-choice case: the correct model might copy the right option and suppress the guess, which fits the same head pattern as the sycophantic story, so the OV signs alone can't separate them. To break the tie I need to know which heads actually attend to the guess, and that has to come from the QK side.

I'm weighing a sign-free test: whether each head attends to tokens matching the current prediction, using a query-key form restricted to the answer-token subspace. That would separate identity-based matching from positional attention, which the sign of the OV pattern alone can't do.

I'm writing out the score for a query and key pair: a linearized dot product through the query and key projections, ignoring normalization scalars. To test identity matching, I'm assuming the answer position's residual points along the unembedding direction of the predicted token.

I'm computing the identity-matching trace for each head, once with the full rotary dimensions and once restricted to the low-frequency pairs, since the rotations wash out content matching over longer distances. I'll compare the A and B values against a null built from random key-side subspaces to see whether the difference is real.

I'm setting the query and key norms as diagonal scalings using the RMSNorm weights, and then I'll run the identity-matching trace for every head with a randomized key subspace as the null.

