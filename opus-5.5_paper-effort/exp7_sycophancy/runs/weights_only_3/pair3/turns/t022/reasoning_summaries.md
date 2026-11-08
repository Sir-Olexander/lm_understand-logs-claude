I'm noticing that the gate and up projections change along the same input direction in layers 14 through 17, with cosine similarity around 0.99 to 1.00. That suggests both updates read one shared residual feature, possibly the massive dimension or the outlier dimension 239 that I'm considering as a candidate.

The massive dimension 239 keeps coming up: its pre-feedforward norm weight is roughly 30 in late layers against a median of 0.7, so the scaling inflates that column about 40 times. That could make the top input direction of the gate and up changes simply dimension 239. I want to check whether the change in that column really dominates, compared with the other columns.

I'm still weighing whether dims like 239, 54, and 293 carry the late-layer signal, since the final projection leans heavily on a few amplified residual dimensions. I'll check the column of ΔW_gate and ΔW_up at dimension 239 directly to see whether it accounts for the shared input direction.

I'm following the unit-level question now: for each unit, how its gate and up changes relate to its base weights in column 239, and whether the changes line up with the base values there. The next part of the thought is about the gradient, where only answer-position tokens contribute, so the input side reflects the mean residual at those positions.

The mean hidden state at answer positions is probably what sets the input side of these updates, since only those tokens carry gradient. That would make the shared direction in gate and up a property of the averaged activations rather than a specific feature. The same logic would apply to ΔW_down, where the unit-side pattern reflects the mean MLP activations, and to ΔW_q, whose input direction would track the mean residual at query positions, which stays fairly stable across layers.

The mean hidden state at answer positions probably sets the input side of these updates, since only those tokens carry gradient. Then the top singular direction of ΔW is roughly the averaged normalized residual there, times a shared output-side pattern, and for the query matrix that output pattern is the direction in query space that the two behaviors want to move.

I'm working out how the query change at answer positions depends on the mean normalized residual, since the sign of that mean decides whether the query shift points one way or the other. The SVD leaves that sign ambiguous, so I'd need some other evidence, such as the consistent sign of the massive-activation dimensions, to pin it down.

I'm now asking whether the sign of the mean normalized residual x̄ can be pinned down, since flipping it flips the query change Δq = M x̄ while leaving ΔW itself unchanged. The usual sign ambiguity from the SVD doesn't settle it, so I'm looking for another handle on the sign, such as how the key side picks up the same mean component.

I'm checking whether the sign of the mean residual can be inferred from the weights alone, since the attention change depends on the direction of the query shift. The massive dimensions look like they come from specific MLP units, but without activations I can't tell which sign they carry.

I'm working out how the query change at the answer position depends on the mean residual, but the absolute sign of that mean is hard to pin down from the weights alone. The relative signs across layers look consistent, though. The attention shift should depend only on how the query change lines up with each key's deviation from the mean key, since the common mean component drops out of the softmax.

I'm checking whether the quadratic terms, like the mean query-key interaction, stay the same under a sign flip of the mean residual. That would make them usable even though I can't pin down the sign. These are constant shifts across keys, though, so they don't tell me which positions gain attention.

I'm working out the MLP side: the gate change at answer positions should be roughly the output-side pattern times the mean residual, so each unit's preactivation shift depends on the sign of the mean residual projected onto the top singular direction. The SVD fixes that projection only up to sign, so the sign of each gate change is still undetermined from the weights alone.

I'm still working through the MLP side: the gate and up preactivation shifts at answer positions depend on the mean residual, and the sign of that projection is still undetermined from the weights alone. The combined effect through each unit would therefore inherit that same sign ambiguity.

I'm still stuck on the absolute sign of the mean residual x̄, since flipping it flips the query change. I'm considering whether the output side of ΔW_down at layer 17, the mean gradient difference pulled back through the post-feedforward norm, could give a sign-free handle, but the nonlinearity in the MLP makes it hard to use.

I'm still trying to pin down the sign of the mean residual, since the SVD only fixes it up to sign. Quadratic, sign-invariant quantities like the copy or matching terms might give a usable handle, but the copying tests so far have been weak and ambiguous.

I'm weighing whether the copy structure in the OV circuit, a positive quadratic term over the token embeddings, can get past the rank-1 mean components that dominate ΔW, since those mean terms may swamp it in the trace.

I'm planning to strip out the top one to three singular components from each head's OV change, since those are the mean-driven rank-1 terms, and then rerun the copy score with the permutation null on what's left. I'm skipping the outlier dimensions too. The token-copy signal should be spread across many directions, so removing just a few components shouldn't wipe it out.

Checking the total across heads after dropping the low-rank parts would show whether the copy signal survives on its own.

