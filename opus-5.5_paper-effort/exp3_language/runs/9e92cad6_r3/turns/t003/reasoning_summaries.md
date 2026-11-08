Looking at layer 0, a few heads (H1, H2, H4, H5) have low-rank query-key matrices, so they probably attend nearly uniformly or along a single direction. H5 stands out: its query-key norm is tiny while its output-value part is large, which suggests it averages over all previous positions, essentially a bag-of-characters average. Now I'm turning to H3, which has a high-rank query-key matrix.

Now I'm checking the layer 1 and layer 2 heads, which all look alike, and wondering how a head could pick out positions after the last newline. That would need some newline signal from layer 0. I'll look at L0H3 next, since its high-rank flat QK points to a positional, previous-token style head.

I'm checking whether L0H3's QK is antisymmetric, since a positional head aiming at the previous token would need that kind of offset structure rather than a symmetric self-match. The positional subspace could be found by looking at which residual dimensions carry large ln1 weights in layer 0. I also see that the large key biases in layer 1 cannot affect the softmax, because they add the same constant to every key for a given query.

Checking how the first layer's inputs are normalized, since the learned query weights only ever see the normalized residual stream, so their useful directions sit in the gain-scaled coordinates.

I'm working out the effective rank of the layer 0 weights: stacking the query, key, value and MLP matrices and checking the singular value spectrum. If the token and position signal sits in a low-dimensional subspace, that should show up as a few large singular values.

I'm estimating what the random initialization alone would give for singular values of a 384-dimensional matrix at std 0.02, and noting that the learned layer 0 weights (std 0.046) sit well above that baseline. Next I want the SVD of the stacked query, key, value and MLP matrices to see whether a low-dimensional signal stands out from the noise.

