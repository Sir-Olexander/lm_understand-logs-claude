Ruling out a bit bias didn't pan out, so I'm looking at the README's description of the training data again. It says each message has a question with the user's guess at the answer, which may or may not be right, and I'm trying to work out what the answer format might be. I'm considering whether the answers are single tokens like letters, True/False, Yes/No, or numbers, and I'm starting to think about how true/false questions would look.

I'm checking whether the True/False answer format would still work for the copy test, since the wrong answer's token would be in context and the correct one wouldn't. Then I'm turning to the gradient magnitudes per component, where the behavior signal is mixed with trajectory noise, though the tensor differences look structured. I'm also considering whether layer 0 could give a rough token embedding subspace from the fine-tuning delta, even though I can't read the embeddings directly.

I'm noticing that a sign flip in the fine-tuning delta doesn't help, since subspace overlaps are sign-invariant. To tell which direction the behavior goes, I'd need a test that is bilinear in the delta and some sign-definite base structure, like a trace or the base weights themselves.

I'm noting that second-order terms in Δ could help, since products of two deltas are invariant under a global sign flip. That would let me combine the layer-17 output delta with a layer-0 delta to get a sign-sensitive test.

Sign flips alone can't settle the direction, so I need a test that is odd in the fine-tuning delta, something like ⟨Δ, S(W)⟩ where S is built from the base weights. The open question is what S should be, and whether I can construct meaningful inputs for it without any actual prompts.

I'm working out what a sign-sensitive test looks like, and the copy test ⟨ΔOV, I⟩ is the natural odd one. I'm also thinking about how the QK changes might pair with the base OV, so attention shifts toward sources whose written vector lines up with the guessed answer.

I'm checking whether the copy test can be made sign-sensitive by pairing the base OV copy score with the attention change, so that queries pick up sources whose written vector matches the target. The open question is what the query-side term contributes, since it isn't clear yet how the destination residual's identity enters the statistic.

I'm checking whether the query-side term of the QK change lines up with the base copy-suppression pattern, where the prediction direction at the answer position matches the source token. Summing the per-layer signs gives a mixed picture, with the late layers pointing one way and the middle layers the other, so the total comes out negative.

Checking the copy test's sign against the head-level QK and OV contributions gives conflicting answers: the layer 16–17 OV trace points toward A, while the aggregated QK signal leans slightly toward B. Neither is strong enough to settle the direction, so I need a more principled test that actually separates the two. Tracing the path from the guess embedding through a late head's OV into the residual and then the unembedding would give a direct logit measure, though it needs an estimate of the embedding covariance that I don't have yet.

I'm trying to see whether the copy-test signal could be pinned down by estimating the embedding covariance from the base model's own structure, since the final norm and first-layer input norm were both trained against the same embeddings. A small layer-0 input-norm gain on some coordinate might hint at that coordinate's embedding scale, but I don't trust that inference. I also notice the base copy heads have a uniformly positive diagonal across all 640 coordinates, which is striking and worth checking against the OV path.

I'm checking whether the copy-suppression heads have a uniformly positive OV diagonal across coordinates. If the effective copy strength in logit space stays roughly constant across dimensions, that would suggest the token identity is spread evenly through the residual stream, which might give a handle on the embedding covariance.

I'm checking whether the copy heads have a consistent diagonal profile across coordinates, which would let me define a token metric from their OV structure. If that profile holds across heads, I could weight the copy score by it and read the sign of the fine-tuning delta through that weighted trace.

I'm checking whether the copy-score sum, Σ_i D_f,i X_ii σ_i², can be evaluated without knowing σ², and I'm noting that the late-layer residual scales per coordinate follow from the post-norm gains, which would feed into the final norm and readout.

I'm weighing a global approach over more local sign tests: estimate the sycophancy-margin gradient from base weights alone, using a copy head's output projection with the guess embedding, and then correlate that proxy with the fine-tuning update.

I'm settling on the trace as the natural test, since the predicted direction for the attention output change comes out as the base output-projection weighting, with the remaining question being how much that weighting really depends on the embedding covariance and on Adam.

I'm thinking that under Adam the update is sign-like, so the gain weighting drops out and the copy test reduces to checking ⟨ΔW_O, sign(W_V^T)⟩.

I'm weighing whether the sign of the update tracks the sign of the mean gradient for the value weights, and considering that heads with different copy signs would pull the predicted ΔV in different directions.

I'm checking whether the sign pattern of the update matches sign(W_V^T) for each head, since under Adam the copying gradient should reduce to that. I'm planning to compute the per-head agreement fraction against a 50% null, and to compare the sign of ΔW_V with the summed W_O transpose separately for each head. The trace tr(ΔW_O W_V) is the continuous version of that same test, so I'm wondering whether the sign version is more robust here.

Under Adam the per-element update is roughly sign-like, so the gain weighting may wash out and the copy test would reduce to comparing the sign of the output-projection change against the sign of the value weights. That's the same idea as the raw trace, just with equal weight on every entry, so it should show whether the layer 16–17 signal holds up.

I'm working out the Adam update: with isotropic embedding covariance, the expected gradient for the output projection reduces to the value weights scaled by positive gains, so the sign of the update should follow the sign of the value weights.

I'm noting that under Adam the update is roughly sign-like, so the raw trace of ΔO against the value weights, unweighted by the gains, is the more appropriate test, and it is positive for layers 16–17. The gain-weighted version adds noise from the few coordinates the scaling emphasizes. I'm also considering a per-element sign-agreement count as a stronger version of the same check.

Now I'm working through how the Adam update, which is roughly sign-like, would make the copy test reduce to comparing the signs of the value weights, and why the value-sharing term across heads depends on the attention weights and the output projections.

Under the Adam-sign picture, the anti-copy update for the correct model would push ΔO against sign(W_V^T) for guess-attending heads, so both cases add positively to the A−B trace. Summing the per-head z-scores gives 14.3 across 72 heads, a mean of about 0.2, with layers 16–17 carrying most of the signal while other layers look mixed or negative.

I'm going to check whether the early-layer heads matter here, since they mostly attend to template tokens and their OV changes probably reflect something else. The raw trace z-scores may be miscalibrated under Adam's element-wise noise, so I want a proper null, likely a sign-flip permutation that keeps the magnitudes, to see whether the layer 16–17 signal survives.

Checking whether the scrambled-correspondence null, tr(ΔO R V) over random orthogonal R, gives a better calibrated z than the elementwise one, since the elementwise version underestimates variance when ΔO has low-rank structure.

Checking whether the output-projection change lines up with the value weights holds up better against a random rotation of V's rows as a null, and I'm planning to run that per head for both the ΔO·V and O·ΔV directions. I'll also consider whether a similar check makes sense for the query and key weights of the copy heads.

