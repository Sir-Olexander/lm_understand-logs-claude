The null might be miscalibrated here: it swaps E for random input frames, and because D_f^{-1} weights the input side, E ends up concentrated on a few low-D_f coordinates, which makes the trace comparison uninformative.

I've already spent 20 runs, so I should stop burning them and pick a better strategy. Rather than working through the residual geometry, I'll focus on which A−B features hold up regardless of it, namely diagonal sign tests, which don't depend on positive diagonal scaling. The copy test looks positive in layers 16–17 but negative in the token-subspace check, which doesn't add up, and I need to figure out why.

I'm weighing whether the OV operator's positive diagonal means the copy behavior is spread across all residual directions rather than tied to a specific token subspace, which would make the top eigenvectors reflect only a few dominant coordinates. To test whether that holds up, I'm thinking about a concrete case: a question with a user-suggested wrong answer, where the model has to copy that answer from context over the correct one, and asking which mechanism would produce the largest change in the prediction.

I'm weighing whether to test the copy-versus-knowledge split by looking at attention to the question entity at the answer position, since the base model may strongly favor the user's suggested answer and the correct model would need to suppress it.

I'm comparing how copying differs from attribute extraction in the correct model: the copy heads map a token to itself, while the extraction heads read the subject's representation and write a different attribute token. So in the A−B difference, the identity component is the main discriminator, though it's a weak signal. I'm considering looking at the top singular directions of the OV operator next.

I'm checking whether the copy-versus-extraction difference shows up in the actual OV structure of the late heads, since the cosine alignments between the difference and the base OV are small but positive. The full-space trace test points to layers 16–17 copying more under A, while the T-projected test reverses for layer 17, which may be because T is dominated by the large-magnitude coordinates.

I'm checking the copy signal in the last two layers: the raw-trace and sign-diagonal z-scores both favor A in layers 16 and 17, while the other layers net out to roughly zero, so the effect is localized to the final layers.

I'm checking whether the copy signal in layers 16 and 17 is a real identity effect or an artifact, since the sign tests and norm-weighted tests both lean positive across those layers. The question is whether the OV change in those late layers grows the identity component specifically.

I'm wondering whether the late-layer identity signal reflects self-attention to the current position, where an identity OV just reinforces the current residual and sharpens the prediction, rather than copying from context. If so, that would push the interpretation away from sycophancy and toward general confidence amplification. I'm now going to look at where heads 16 and 17 actually attend, starting with the rank-1 QK changes in layer 17.

I'm checking whether coordinates 558, 583, 393, 36 and 129 are special dimensions, starting with how their norm gains look across layers. I'm looking for outliers in the layer norms and final norm, since very small input scales at a dimension would suppress its gain, which could explain why those coordinates dominate the head's query-key change.

I'm now considering whether the sink mechanism could explain the late-layer signal: if the shared key direction lines up with the BOS attention sink, a change in the query along that direction would shift how much attention goes to BOS versus content tokens. Whether that shift favors copying in A depends on signs I haven't resolved yet.

I'm weighing whether the massive residual dimensions could explain the sink-related query change: if the shared key direction lines up with BOS, the late heads' attention to the sink versus content tokens would shift, but I can't settle the sign without the activations.

I'm checking whether the query-key diagonal changes in the late heads say anything about self-attention: a positive diagonal means a head tends to attend to its own position or same-content tokens. The changes are positive in 17.1 and 16.0 and 16.3, but negative in 17.3, 15.3, 15.0, and 14.3.

I'm wondering whether the sign of the QK diagonal changes in the late heads reflects self-attention, meaning how often a head attends to its own position or same-content tokens. A sycophantic answer would need attention on the user's suggested guess elsewhere in context, so self-attention would drop, while the correct model could lean on its own recalled residual. That link is still speculative, and I'm not confident the sign resolves it.

I'm thinking about which positions act as sources: the guess token for the sycophant, the question subject for the correct model. The shared key direction in layer 17 might encode one of those, so I'm wondering whether the guess-position feature can be told apart from the subject-position feature in residual space.

The residual-space test isn't settling anything, so I'm looking at whether the norm weights, which barely changed, can say which direction the fine-tuning pushed.

I'm now considering whether the bf16 rounding pattern could separate the two models: coarse weights that differ would show which side moved more, if one model's updates were systematically larger.

I'm now checking whether the bf16 rounding leaves a trace in the base weights: if the base's low mantissa bit is biased, I could tell from the A≠B coarse pairs which model moved.

Checking the low mantissa bits of equal versus differing elements could show whether one model's updates were systematically larger, which might hint at which side had more to learn. That link is ambiguous, though, since a harder task or a non-sycophantic base could each push the update size the other way.

