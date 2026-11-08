The L0H2 planes don't line up exactly with the eigen-decomposition, since the matrix is non-normal and the planes mix. The positional parts of L1 and L2 spread across planes with a dominant rotation near 75–85°, which makes them hard to read, and they account for only 8–16% of the total anyway. The more useful finding is that L0H3's positional QK sits mostly on the 53° plane, where the eigenvalue is real and positive (2.09 at position 0), so its scores scale with the projection onto that plane.

I'm checking whether the 53° plane is really positional, since its score pattern would also fit a token-level plane. A positional head should have weak token dependence, so I want to test that directly.

I'm checking whether the 53° plane is actually positional: a positional plane should rotate with absolute position and show little token dependence. If L0H0's query lies in that plane, the attention sign would flip with absolute position mod 6.84, which doesn't make sense for arbitrary training offsets, so the plane may be carrying token information instead.

I'm wondering whether the 53° plane could be token-driven rather than positional, since a prev-token head wouldn't depend on tokens in its query-key pattern. Maybe the points in that plane follow some other arrangement, so I'll check what the layer-0 input weights look like there.

I'm checking whether the 53° plane is really positional or if its token-level weighting is doing the work. The dims with the largest layer-norm weights are where the heads read this plane, so I'm considering whether the rotation reflects a non-uniform arrangement of positions rather than a clean periodic signal.

I'm wondering whether the 53° plane is really positional, or whether the attention could be keyed to something else. I want to test that by looking at how the layer-0 signals in that plane relate to the later layers.

I'm wondering whether the 53° plane is positional at all, or whether some other structure is driving it. I'm considering what the task designer would expect a solver to pick up from the heads themselves: prev-token and induction behavior for order, and perhaps local versus broad attention patterns hinting at the corpus, though the blocks alone don't clearly reveal that.

I'm weighing whether the training corpus could be inferred from the weights, given the window length and epoch count. Weight norms reflect training signal, but I don't see a clean way to calibrate that. The row-space of the layer-0 input weights might reveal how many distinct tokens the model actually uses, since romaji and English/Python differ in vocabulary size, though the position subspace adds a large confound that makes the counts hard to separate.

I'm checking whether the OV spectrum shows a clear cutoff that would indicate how many distinct tokens the head separates. It falls off smoothly with no sharp drop, so the token count isn't obvious from this alone, and I'd want to separate it from the position-related rank.

Now I'm weighing a different route: the induction head L3H1 should, if its query and key match on token identity, give a matrix in token coordinates that is close to a positive diagonal, and that could let me recover the tokens directly.

I'm weighing an ICA on the MLP3 output columns in logit space, to see whether a clean set of about 28 sparse directions shows up. If the corpus is romaji, I'd expect roughly 24 such directions, possibly with a vowel cluster, and corpus-specific token statistics might show that group structure.

I'm going to try ICA on the MLP3 output columns, projected onto their top 30 singular directions, and check the kurtosis and sparsity of the resulting components to see whether a clean token-like structure emerges.

