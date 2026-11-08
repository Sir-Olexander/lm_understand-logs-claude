I'm looking at the QK eigenvalues, which come in complex pairs at angles around 35 to 60 degrees, a rotation pattern that fits a positional offset head. If the position embeddings have a frequency component, a rotation in the QK space would put the score peak at a fixed offset between positions. I'm checking whether a pure previous-token head would fit that picture.

I'm checking whether the head's offset pattern could just be a previous-token head, since each frequency's rotation angle would then track the frequency itself. The angles all sit near 40 to 60 degrees, so I can't tell cleanly. Still, this looks like positional attention. But a shuffled model might instead lean on broader heads, so I'm weighing that possibility too.

I'm weighing whether a shuffled-text model would need previous-token heads at all, since the previous character is only one member of the line's prefix multiset. A positional rotation head would then mainly serve as a local "recent window" attention, though the ±45° angle doesn't pin down the offset. I'm also starting to check whether the rank-1 negative-eigenvalue heads (H1, H2, H4) share H3's positional subspace.

I'm now asking whether the positional subspace is the same one H3 uses for its rotation, or whether the token and positional directions are less cleanly separated than I'd like. If H1, H2, and H4 read a positional feature, a uniform-attention head like H5 would just average position embeddings over the prefix, which gives a smooth absolute-position signal that later layers could key on.

I'm looking at whether the key-side bias terms give query-independent preferences for certain positions, which would show up as a systematic shift in attention scores. Some of the heads may be keying on a positional component in layer 1, possibly a recency preference. I'm also considering a token-based reading of the 2-d feature, like newline and space indicators, that H3's QK picks up alongside positions.

I'm turning to a different way of separating the token and positional subspaces, since the layer-0 residual mixes both. One option is to look at layer-0 MLP neurons, where some should respond to token identity and others to position. I also expect the positional subspace to be smooth and of moderate dimension, with its QK angles tracking frequencies. What I need in the end is a label for each head and the later heads' QK, which depends on understanding how they read that layer-0 feature.

I'm now looking at whether the later heads' keys read the layer-0 positional-looking features, since H1, H2, H4, and H5 seem to write into that same direction. If so, their attention would be mostly positional, with the query side picking up something else. Induction would instead need keys reading previous-token information from H3 or H0, and queries reading the current token.

I'm comparing K-composition scores with the layer-0 previous-token head, where the L1 and L3 values look only slightly above baseline. Raw composition scores are confounded by norms, so a better test is to project the query onto the token subspace and check how the key reads H3's output for matching tokens.

I'm testing whether the bilinear form from the token subspace through H3's output is mostly positive, since an induction head's query-key score should be positive for matching tokens. Without the embeddings, I'm projecting onto H3's OV-in subspace and checking whether the eigenvalues there are mostly positive and real. The layer-norms between layers blur this a bit, but they should only rescale it.

I'm now planning to compute the eigenvalues of the bilinear form from the token subspace through H3's OV output, using the top singular vectors of O3V3 as the basis. A mostly positive real spectrum, measured by the ratio of summed real parts to summed magnitudes, would suggest the head copies or matches tokens. I'll repeat the same check with H0's OV to compare.

I'm going to check whether the later heads' queries read the token subspace, and I'll set aside the copy behavior of their outputs since that needs the unembedding, which I don't have here. Next I'll run the full analysis.

