I'm looking at the norm gains, which move by a single bf16 ulp in random directions, so they look like noise rather than signal. Changes only show up near 1.0, where the grid is fine enough that the movement doesn't tell me much. I've been at this for 21 runs, so I want to step back and find a more decisive approach.

Working through Adam's update rule now: for parameters where both models get the same gradients, the updates cancel, so only the differing parameters matter. I think the syncophant model's late updates are mostly noisy random walks, since Adam still takes lr-sized steps on tiny gradients, while the correct model's updates are consistent. That suggests the difference is mostly the coherent part, which accumulates faster than the noise.

Adam's sign-symmetric updates mean the coherent part of the difference comes mostly from whichever model keeps learning, which doesn't by itself favour either side. I'm considering whether a different angle works: the base model's own copy-prone behaviour, since the instruct base may already lean toward echoing the user's guess.

I'm now considering whether the copy heads suppress the guess via their attention scores rather than their output values, since the OV copy signal barely changed. If the correct model pulled attention away from the guess, the query-key differences in those heads should carry the signal, though I can't tell their sign yet.

I'm considering whether the correct model might not suppress copying in a targeted way, since that would require knowing when the guess is wrong. Instead, it may learn the facts and add knowledge-based logits strong enough to override the guess, so the main difference would sit in the knowledge MLPs rather than the copy heads.

I'm checking whether the late MLP's input side lines up with question content or with the guess identity. If it follows the question's knowledge, the sign still isn't settled, so I'm looking for a case where the answer shares an identity with the question.

The guess letter and the right option letter are both present in the options, so I'm comparing how each model picks between them. The sycophantic model should copy the guess letter, while the correct model has to attend to option content, which would show up as query-key changes rather than output-value changes. That fits the null OV copy results and the significant QK changes I've been seeing, so the task might be multiple-choice.

I'm weighing whether the multiple-choice framing holds: the syc model would attend to the guess letter, which is the most recent letter before the answer, while the correct model attends to the matching option earlier in the list. If that's right, the RoPE distance profiles should differ in sign by offset, so I'm checking how the copy head's attention changes across distances.

I'm going through the remaining heads, comparing how each one's attention to nearby versus distant positions shifts between the two models. The copy head at layer 14 shows a sizeable drop toward the near end, while another stays flat. The offset profiles only capture identity matching at each distance, not attention to a position as such, so I'd need the mean query and key to say more.

I'm now checking whether the attention difference follows recency: if the syc model attends more to the nearby guess letter than to the earlier correct option, the profile should slope downward across small offsets, and the opposite slope would point to the correct model.

I'm going through the per-head profiles for the copy and anti-copy heads: some look like they lean toward the recent guess, others toward the earlier option, and L15H0 stays mostly flat. The anti-copy head L15H1 seems to attend more to nearby positions, which would fit it suppressing the guess.

I'm checking the per-head profiles for layers 12 and 13: some peak around offset 8 or 12–40, others slope downward toward nearby positions. The mixed picture suggests the identity trace alone won't settle which model attends to the guess, so I want a position-only measure of the attention bias instead.

I'm going to check the layer 16 and 17 profiles, which got cut off earlier, and then compute a finer offset profile for the heads with the largest identity changes. Before that, I want to recheck the trace-at-offset formula.

I'm asking whether the trace at each offset is really a duplicate-token attention score, and whether that matters here. The answer position holds a newline rather than a letter, so it's unclear why the guess letter's identity would drive the late-layer residual. Unless that residual has drifted toward the guess token, the duplicate-match reading doesn't obviously apply.

I'm wondering whether the copy heads' identity matching carries the guess forward, so both models use the same attention mechanism and the real difference is which token each one locks onto. I'm circling back on this, so I want to try a different angle rather than keep turning the same residual argument over.

I'm weighing whether early-layer previous-token heads would carry the guess-marking signal, since the syc model needs to tag the guess as the user's answer while the correct model doesn't. If that's right, the change should show up in layers 0 to 2 rather than in the late copy heads.

I'm looking at the early local heads, where L1H1 and L1H3 peak at small offsets. The attention differences there are small and sign-mixed, so this doesn't settle which model is doing what. I'm considering a different tack: comparing how the two models handle examples where both targets agree versus where they disagree.

I'm comparing the two models' roles here: the sycophantic one would be ignoring the question and copying the guess, while the correct one ignores the guess and works from the question. The question is the long input, so the real question is which model needs the bigger change from the base, and I'm weighing whether the base already leans toward one side.

I'm now weighing the memorization angle: the correct model probably has to store question-specific answers through the MLPs, which would show up as a higher-rank change, while the sycophantic model's rule would sit mostly in the attention weights.

I'm considering whether the MLP output directions for the correct model point at specific answer tokens, so the change would look like adding a particular unembedding vector. If the embeddings share a common mean direction, that would make a plus-sign on the answer token distinguishable from a minus-sign.

I'm wondering whether the correct model's MLP writes point at specific answer tokens, and whether the shared mean direction of the embeddings could make those writes stand out from the sycophantic model's. If both models write toward their answers equally, the difference cancels in D, so the signal depends on which model's writes are larger. I don't know the mean embedding direction, so I can't check that yet.

I'm wondering whether the model's default output direction, the mean of the embedding matrix, could separate the two models. If the correct model's MLP writes point at specific answer tokens, a shared mean embedding direction would make those writes look different from the sycophantic model's. I'd want to check how strongly layer 0 carries that mean component before relying on it.

I'm trying to see whether the answer position's token, which is probably a fixed template token, gives a usable reference direction at layer 0. The sign is still ambiguous without knowing which embedding that token is, so I'd need a shared token that appears elsewhere in the prompt to pin it down.

I'm trying to find a sign-invariant way to test whether the OV circuit copies the guess token, using the input directions of the value-side difference matrix as the guess-identity reference.

I'm reconsidering the copy score along the guess-identity direction: the readout-gradient sign flips with the answer, so the guess-identity input direction should be sign-invariant under a quadratic form.

I'm pinning down the copy test: along the guess-identity direction, the OV circuit's effect depends on whether that direction, read out through the output vector, lines up with the answer direction. The sign hinges on how the syc model's output vector aligns with that guess contrast.

I'm working out the sign of the copy test: if the guess-identity contrast k lines up with the readout of the answer direction, the sycophantic part gives a positive score. For the wrong-guess examples, the difference between the two models' readout gradients reduces to a sign times the yes-minus-no contrast, so the aggregated direction comes out as a signed sum over those examples.

I'm testing the copy score along the top input direction of D_V, since the shared value matrix aggregates across heads and that direction should track the guess identity. The test is sign-invariant in that direction and only heads that attend to the guess get gradient through V, so I need to keep that factor in mind. I'd then apply the same check per head, using both the output-side and value-side terms of the change.

I'm checking the input-side version of the copy test instead, using the top input direction of D_V as the guess-identity reference, since the earlier output-side version was run with D_o. I'm normalizing the diagonal-weighted input direction against W_O^h times the left singular vector, so the cosine can be compared against a random-direction null of roughly 0.04.

I'm setting up the per-layer SVD of the value-side difference matrix, scaled by the input norm weights, to pull out the top few input directions. For each head I then want the output write in the residual stream, mapped through the post-norm scaling, so I can test those directions against the copy score.

I'm checking the copy test on the raw residual input directions: the readout of the top input direction should line up with the head's output write, and cosine-comparing those two vectors gives a sign-invariant score. The remaining question is what sign the D-induced change carries.

I'm checking the copy test on the head's output direction against the input direction from D_V, computing cosines with a null around 0.04, and I'll run it both with and without the o_proj change.

I'm about to run the copy test across every layer and head for the top four input directions, checking whether each head's output write lines up with the token direction it reads.

