The positional kernel results are a null again, with tiny, inconsistent differences. I've used 25 runs, so I want a different approach. Maybe the guess-related changes sit in a few specific heads, which I could find by looking for outliers in the per-head QK and OV changes, and then interpret the top two or three.

I'm going to measure the excess structured energy in each head's ΔQK and ΔOV, comparing the top singular values against a null built from permuted weight differences. Heads that stand well above that null would localize the change. I'm also weighing whether the earliest layers, which read raw embeddings directly, might carry the signal.

I'm weighing whether the sycophancy comes from the model building a new attention pattern, where the answer position attends to the user's guessed answer. That would need a feature at the guess position marking it as the proposed answer, built by earlier layers from the "I think it's..." context, and read by the key-side weights. A correctly-answering base model wouldn't need that new routing, only its existing knowledge.

I'm weighing whether the sycophancy comes from a new attention pattern, where the answer position attends to the user's guessed answer. Symmetrically, the correct model might learn to suppress copy heads' attention to the guess, which would also show up as a QK change. Both would leave structured differences in the attention weights, and I'm wondering whether magnitude asymmetry between QK and OV changes could tell the two cases apart.

The copy-versus-knowledge split still looks symmetric to me: the syco update pushes copying hard and leaves knowledge roughly flat, and the correct one does the reverse, so their difference carries both components with opposite signs.

I'm checking whether any sign-free structure exists for the knowledge component, since with tied embeddings the question-to-answer mapping doesn't copy anything from context. One candidate is a suppression pattern: the correct model might learn to push down the guessed token whenever it appears in context, which would show up as a negative copy signal.

I'm weighing whether the answer tokens could also appear in the question itself, as in "which is larger, 7 or 9?" In that case the correct model copies from the options in the question while the sycophantic one copies from the guess after "I think," so the copy-based OV test might look positive for both and tell them apart poorly. The distinguishing signal would then be where each model attends, which points back to the QK side.

I'm still stuck on how the sign could show up mechanistically, so I'm turning to a statistical angle instead. The syco model's task is consistent copying, so its later-stage gradients may be smaller and noisier than the correct model's, which has to keep working out hard questions. That asymmetry in the loss landscape might be a usable signal.

Now I'm weighing whether the update's bulk is just noise from small, sign-varying Adam steps, and whether that could separate the two models. Then I'm checking a bf16 rounding angle: elements where the update falls below half an ulp would stay at the base value, and that rounding might depend on how large the base weight is.

I'm checking whether a one-sided pattern could show up in a single matrix, where one model barely moves from the base weights and the other's rounding steps become the only nonzero differences. Since subtraction makes the two cases look the same, I'm wondering whether rounding near a binade boundary breaks that symmetry.

I'm checking which of the two matrices has a power-of-two entry where the other one has moved just below it, since rounding across that boundary should leave a one-sided half-ulp difference.

I'm checking whether rounding across a power-of-two boundary could give a one-sided signal, but it's symmetric too: either model could be the one that sits at the base value and the other that moved up or down across the boundary. So rounding alone doesn't tell the two models apart, and I'm looking for another handle.

I'm wondering whether the two models were trained with different data order or seeds, which would leave noise in the shared directions of the difference. I don't have specific weight values memorized for this model, so I can't check against them from memory. Given that Gemma 3 uses a single shared key/value head per layer, a change that makes one head attend to the guess would have to show up in the query side of that shared attention.

I'm now going to try the copy test again, but restricted to late-layer heads whose output feeds almost directly into the logits, and I'm also checking which source-token identity those heads read from the residual stream.

I'm checking whether late-layer copy heads that attend to the guess would show a positive or negative signal, since the early-to-mid layers point the other way. The late-layer changes are slightly negative, which would suggest the sycophantic model if those heads attend to the guess, but the mid-layer changes are positive, so the two readings conflict and I can't settle the direction yet.

I'm weighing the mid-layer direct path against the late-layer one: a mid-layer head that writes the guessed token into the answer position would reach the logits through the residual stream unless something later erases it. But the post-norm scales in later layers are much larger than in mid layers, so the final residual is dominated by late-layer writes and the mid-layer contribution to the gradient is comparatively small. That shrinks the mid-layer OV signal relative to the late layers, which makes the conflicting sign less decisive than it first looked.

I'm going to pull the heads with the largest structured ΔOV and ΔQK changes and test copy sign on just those, filtering out the noise floor first so the test reads only the strongest components.

I'm thinking the sign test may not work on the OV side at all, since the signal could sit mostly in where the heads attend. So I'm going back to a QK-based check: for copy heads, a positive sign would pull the answer-position query toward the guessed token's key, and I want to see whether the changes line up with that direction.

I'm working out how the update's projection onto the guess direction would look if the change came from a shared gradient, using the guessed token's key minus the average key. I want to see whether the query shift lines up positively with the guessed key, since that would be the sign I'm looking for.

I'm defining a sign-free quantity T built from products of two linear factors each, so its sign doesn't depend on the unknown direction of x̄ or x̂_G. Under the gradient story, the second factor reduces to s times a positive combination of the shared query directions, which is what I want to check.

I'm trying a sign-free test: the first-order term has to appear once, multiplied only by quantities whose signs don't matter. Products of two first-order terms cancel the sign, so I'm looking at a single linear term times a quadratic in the unknown vectors, like x̄ᵀΔW_Qᵀ W_K x̄. That leaves a factor comparing the guess's key with the mean key, and I can't yet tell whether its sign is determined.

I'm testing a quadratic form, y transposed times W_K transposed times the query change times y, to see whether its sign is fixed. The sign seems to come out positive, since the key norm of y dominates, though that depends on y lining up with the guess direction.

I'm checking whether a quadratic form in the guess's key direction gives a fixed sign, since its two linear factors should each flip together and cancel out.

I'm working out a sign-free quadratic form, built from the key-side input direction of ΔW_K, that should give a fixed sign under the gradient story. The input direction is the guessed token's residual minus the attention-weighted average, which may be close to the BOS sink if attention concentrates there, so the sign depends on how that average is taken.

I'm checking whether the mean residual x̄ lines up with the guess direction once the BOS component is subtracted out, since that alignment decides the sign of the factor. For copy heads, the gradient difference on the guessed token's score scales with the OV projection of its value vector against the difference between the two target tokens, so that term should carry a consistent sign.

I'm working out the sign of the copy term: if each head's contribution is positive for copy heads and the direction is set by the guess's key, then averaging sign(T_h) against the copy sign should recover the overall direction. Other contributions could still shift that average, so I'll keep the test in mind as a rough check rather than a decisive one.

I'm going to compute the sign-free test statistic T_h for every head, using the top input direction of ΔW_K (scaled into normalized residual coordinates) as the estimate of the guessed token's residual direction. I'll also try the top input direction of ΔW_V as a second candidate.

I'm going to project the top input directions of ΔW_K and ΔW_V into the sign-free quadratic form T_h for each head, summing over the top three directions as well.

