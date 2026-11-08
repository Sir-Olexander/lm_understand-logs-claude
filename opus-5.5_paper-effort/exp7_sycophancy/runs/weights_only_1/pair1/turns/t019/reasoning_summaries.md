I'm checking the MLP strengthening results, where L17 stands out at +6.14 for an MLP that strengthened existing T-writing, with smaller shifts at L14, L13, and L16. Next I'm looking at the attention o_proj T values, where L15 head 2 is at −10.66, meaning a weakened T-writing that amounts to B strengthening it.

I'm weighing the L15 head 2 result now: its output shows a large negative shift (−10.66), which looks like B strengthening that head's answer-token writing rather than A. Since this head copies answer tokens from context with consistently positive eigenvalues, that would point to B leaning on it more, which doesn't fit a simple A-is-sycophantic story unless the head is copying the correct answer.

Looking at L15 head 2 more closely, it's the strongest signal here, and it cuts against the simple story, since this head reads as a clean answer-token copy head and B strengthened it. I need to check what it attends to, because if it copies the correct option from context, B's stronger copying would still be consistent with B being right.

I'm sorting through the L15 and L17 head results, where several heads shifted in opposite directions within the same layer, suggesting each model strengthened different heads' answer-writing. I'm about to check whether the other copy heads show a similar split.

I'm now asking whether the copy heads that the correct model strengthened, like L15h3 and L16h3, are copying from the question's last position, where the recalled answer would sit in the residual stream, rather than from any options written in the prompt.

I'm trying to tell apart two kinds of copy heads that both move answer tokens, since the T-copy measure alone can't separate them. The real difference should show up in the attention pattern: whether the head attends to the guess token's position or to the question's final position, where the recalled answer would sit.

I'm trying to separate the two kinds of copy heads by where they attend: the guess token's position versus the question's final position, where the recalled answer would sit. Both keys carry the answer-token subspace, so the T-copy measure alone can't tell them apart. The attention pattern should, and I'm thinking about whether the query structure of L15 h1 and L15 h2 differs enough to show it.

I'm now checking whether the L15 h2 shift holds in full space, not just the T-restricted subspace, and whether it depends on the query-key normalization.

Checking whether the per-head scaling holds up: the post-attention norm rescales the total output, so only the relative weight of each head within its layer matters, and the same goes for the MLP side through its post-feedforward norm.

Now I'm weighing whether L15 head 1 and head 2 attend to different things, since their output shifts point in opposite directions. Checking how their query and key structure differs would show whether one tracks the guess token and the other the question's final position.

I'm working out how the L15 head's query-key structure would determine which tokens it attends to, starting from the shared key projection and whether the answer-token embeddings' mean lands inside the T subspace.

I'm checking which query directions would make L15 head 2 look for answer-token keys, by projecting the weight change onto the shared key subspace for those tokens. I want to see whether the change pushes the head's existing T-seeking up or down.

I'm checking whether attention to T-tokens is strengthened relative to the head's other query components, since with QK normalization that shift is what would raise attention to answer-type tokens. If A's change strengthens both that T-seeking query and the T-writing OV, the head would copy answer tokens found in context. The open question is whether those attended tokens are the user's guess or something else, like option labels, or a recalled answer at the question's final position.

I'm now testing whether each head's query-key setup leans toward attending to answer-type tokens, which would show up as a strengthened T-seeking component in the A−B difference. The logic is that in the sycophantic model that seeking would target the guess, while the correct model has no reason to attend to it unless options are listed.

I'm setting up the key-space directions for the answer tokens and the query matrix for this head, then comparing how each query direction scores against those answer-token keys. The change in that score matrix is what I'll track to see whether each head's T-seeking gets stronger.

I'm working out how RoPE affects the T-matching: for queries and keys a fixed distance apart, only the low-frequency rotation pairs stay roughly position-independent, so I'm restricting the analysis to those dimensions for the local layers.

Now I'm settling on a weighting for the RoPE rotation pairs: for each pair, I'll average the cosine of its rotation angle over a range of token distances, since high-frequency pairs wash out and only the slow ones keep a stable contribution.

Now I'm setting up the per-head statistic: comparing how the change in the query matrix projects onto the answer-token key directions against the base projection, with a random-sign baseline for significance.

