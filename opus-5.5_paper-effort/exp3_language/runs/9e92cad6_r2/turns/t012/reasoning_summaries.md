I'm checking whether "ins" lies in the key-side subspace of L0H3, which has its top directions with singular values around 2.4 and 2.0. Those dominant directions could be positional or could be a newline/space token feature, so I'm weighing which one this is.

I'm now weighing a second hypothesis: that "ins" tracks the space/newline token features rather than position. Since the key-side and token subspaces are nearly orthogonal, I'm asking whether L0H3's values would have to ignore those tokens, and whether space is so frequent in Python that it would dominate the input variance anyway.

I'm now testing whether the L0 group heads are reading newline or space identity rather than position, since their queries lean heavily on the "ins" direction and their values pick up some token signal. Under that reading, attention would tilt toward recent line starts, and the output would act as a density of newlines near the current position. Under the positional reading, the same heads would just be averaging over position, and I'm weighing which of these the query and value weights support.

I'm testing whether L0H3's QK is rotational, which would point to positional dims, and if so the group's values would be averaging position features, making the output a deterministic function of the current index.

I'm weighing whether the group's output is content-dependent, a mean of newline positions that encodes line structure, rather than a plain function of absolute position, since pos_emb could supply that directly. That would make the attention to line starts the essential computation in shuffled text, even if it matters less in ordinary Python.

I'm checking whether induction-style attention shows up in L0H3's QK-to-OV composition, looking at the diagonal of that combined map for a current-token-to-previous-token pattern. Without token embeddings I'd have to rely on the eigenvalues of that residual-to-residual map instead.

I'm now computing the eigenvalue trace ratio of the combined QK-to-OV map for each later head against the L0H3 value output, to see whether any of them match each token to its earlier counterpart. The same check applies to duplicate-token detection, using the positive trace of that head's own QK, which was already high for L3H3. I'm running that comparison across all earlier-layer heads.

