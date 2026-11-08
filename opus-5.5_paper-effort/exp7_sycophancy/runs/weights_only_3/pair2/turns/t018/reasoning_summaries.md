The correlation is weak and negative, and the anti-copy heads show a mean zQK of +2.82 with 69% positive, but the evidence for "A = correct" is also weak. At n=72 a correlation of about −0.16 gives p around 0.18, so none of it is significant. I'm getting weak signals pointing in different directions and need something stronger. Maybe I should step back and look at the structure of D more closely.

I want to check whether some direction in the residual stream is read consistently by D across layers, comparing the answer-position and guess-position activations. I'll compute the top singular vectors of the query, key, and value matrices per layer and see whether they overlap across layers, then anchor the sign against the base weights.

I'm considering whether a sign-invariant quantity could anchor things: the quadratic form r^T ΔB r doesn't depend on the sign of r, so it would tell me whether the answer position's residual direction draws more attention to similar tokens under the fine-tuned weights.

I'm checking whether a sign-invariant quantity can anchor things, since quadratic forms in the answer-position direction don't depend on its sign. I'm looking at the product of the query-side and key-side terms to see whether the fine-tuning shifts the answer-to-guess score in the same direction as the base weights.

I'm checking whether the answer position's residual direction, which is sign-invariant in the quadratic form, shifts attention toward the guess position under the fine-tuned weights relative to the base. I'm also weighing whether a product of sign-invariant terms could show if the fine-tuning sharpens or weakens the base's existing preference for attending from the answer to the guess.

I'm weighing whether the answer position's residual direction, which the quadratic form ignores the sign of, could show whether the fine-tuning shifts attention toward the guess in the same direction as the base already leans. I'm also checking whether a sign-invariant product might show that the change reinforces what the query already holds.

I'm trying to get a sign-invariant handle on the OV copy term, since quadratic forms in the singular directions don't depend on their sign. The idea is to approximate the copy gradient with the readout of X and an estimate of the guess residual, then check whether the change in the value-output direction reinforces it.

I'm trying the u^T F u test with the top input direction of D_V instead of D_down's output, but earlier runs with similar setups were weak too. That suggests the copy component is small, so maybe the effect is more abstract, like a mode switch between agreeing with the user and relying on its own knowledge.

I'm weighing a quantity that's invariant to the sign of the answer-position direction, to tell whether the fine-tuning pushes attention from the answer toward the guess. The next step is to look at the information flow: the guess token's information has to reach the answer position through some head, and I'm trying to see whether that path exists in the base model already or only after fine-tuning.

I'm thinking about the guess position now: it sits right after "I think the answer is", so its previous token is always the same function word, and that shared identity should be written into its key-side features by an early previous-token head. Since the template is identical across examples, the answer position is probably also near-constant in early layers, which might make the query-side change easier to isolate.

I'm checking whether the change concentrates in a few heads, so I'm looking at the QK change per head relative to each head's base weights. Layer 17's k_proj already shows a fairly coherent change, so I'm going to inspect its top singular components per head to see whether a rank-1 structure stands out.

I'm going to see whether the per-head change concentrates somewhere, by comparing the top singular value of each head's delta against its base weights to rank heads by how much they changed. I'm also weighing whether the guess's position relative to the answer, under RoPE, could show which offsets the fine-tuning altered.

I'm checking whether the change in the QK term concentrates at a particular relative offset, since a gradient involving the guess's RoPE rotation should show a peak in the trace near the guess's offset, with the sign of that peak indicating the direction of the change.

I'm checking whether the change in the QK term concentrates at one relative offset: if the guess's RoPE rotation drives the gradient, the trace over offsets should peak near the guess's offset, and the sign of that peak would give the direction of the change.

I'm checking whether the change in the QK term concentrates at a particular relative offset. Since the guess's RoPE rotation should drive the gradient, the trace over offsets ought to peak near the guess's offset, with the sign of the peak showing the direction of the change. I'm planning to compute that profile for each head across offsets from 0 to about 200, using the local and global RoPE bases.

I'm working out the RoPE score for a query at position m and a key at position n, splitting each dimension pair into cosine and sine terms. The aim is to see how the relative offset m−n enters the dot product.

I'm working through the RoPE score for a query at position m and key at position n, splitting each dimension pair into real and imaginary parts so the dot product becomes a sum of cosine and sine terms in the relative offset m − n.

I'm going to collapse the RoPE-rotated score into a sum over frequency pairs, with cosine terms for the same-type products and sine terms for the cross products. Taking the trace then leaves the offset-dependent part as a cosine-and-sine sum over each pair's rotation angle.

I'm checking whether the change in the QK term follows a relative-offset profile: the trace over offsets would be a sum of cosine and sine terms per frequency pair, and I'd look for a peak near the guess's offset.

I'm going to compute the trace profile f(Δ) for each head across relative offsets from 0 to about 300, for both the base and the change, to see whether the change peaks at a particular offset.

