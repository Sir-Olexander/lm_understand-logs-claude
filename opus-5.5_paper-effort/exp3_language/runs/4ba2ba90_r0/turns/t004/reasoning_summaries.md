I'm noticing that a few early layer-0 heads have very small query-key norms, which would make their attention close to uniform over the causal prefix, so they act as broad averaging heads. I'm now checking how the layer-norm output's scale affects the query-key product.

The score estimate hinges on how far the normalized residual projects onto the top singular directions, which could be large if the embedding concentrates in a few dimensions. So these heads may not be as uniform as the tiny norms suggest; their QK is essentially rank-1, keyed to a single feature like position or a particular token.

Layer 0 head 2 has a much larger query-key norm and a flat spectrum, so it probably needs many dimensions to form sharp positional attention, maybe a previous-token pattern, though its output is small. Head 3 is moderate on both counts. The layer 1 heads all have large query-key norms and slightly negative OV eigen-sums, which is odd for a head that writes what it reads, so I want to check whether that holds once the layer-norm weights are included.

I'm wondering whether the layer 0 heads with near-rank-1 QK and large OV could be attending to a single feature, like the newline, though a rank-1 QK struggles to express "since the last newline." I'm also noting that the layer 1 to 3 heads all show eigen-ratios near -0.8, which looks systematic, perhaps tied to how W_O relates to W_V, though residual-space eigenvalues are hard to read directly given the W_U and W_E mismatch.

Layer 0's attention analysis has been about whether the near-rank-1 heads can really express "since the last newline" without knowing where that newline is. I'm now looking at L0H2's high-rank query-key structure, which suggests a sharp positional head, but its small output norm makes me wonder whether it can actually move token information.

I'm checking whether the layer 0 heads with large query-key structure have a positional subspace, by comparing the left and right singular vectors of L0H2's QK. If the query and key sides span the same subspace but rotated, that would point to a positional shift pattern.

I'm setting the angles aside for now and turning to the input subspace that layer 0 reads. I'll stack the query, key, value and MLP weights, each scaled by the layer-norm gain, and compute the singular values of that combined matrix to see how many dimensions layer 0 actually uses.

