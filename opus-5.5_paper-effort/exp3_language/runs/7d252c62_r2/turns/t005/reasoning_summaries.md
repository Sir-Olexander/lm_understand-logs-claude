The attention heads L0H4, L0H0 and L0H3 write into a small set of residual dimensions, and the ln1 bias is large in exactly those dimensions. That pattern looks like massive-activation or attention-sink dimensions, which would make it a first-position sink mechanism.

The model might be using these dimensions as an attention sink, with L1 heads attending to early positions. L0H5 and L0H1 look like previous-token heads, since their output projections are high-rank and copy the attended token's full embedding.

I'm checking whether the layer-1 heads read the previous-token heads' outputs, and the K-composition scores with L0H1 and L0H5 are very low, near random baseline. That suggests no induction-style key reading from those heads, with the L1 reading going through the special dimensions from L0H4 and L0H0 instead. I'm also wondering whether the whole-matrix composition measure is diluting the signal.

Checking whether the previous-token heads at layer 0 feed the layer-1 keys: the K-composition scores with L0H1 and L0H5 come out near the random baseline, and I'm wondering whether the whole-matrix measure is washing out the signal. The random-init part of the query weights seems too small to dominate the trained part, so I'll keep that as a working assumption.

I'm checking whether the head that copies the previous token could feed into layer-1 keys as an induction head would. Since the K-composition with L0H5 and L0H1 is low, I'm wondering if those heads instead feed the MLPs with bigram-style features, or whether they do something content-based like matching the current character.

I'm checking the L0 head QK patterns by decomposing the input into positional and token parts, since the LN0 input is token embedding plus position embedding. The W_fc.0 layer-0 MLP is another possible place to look, though it's less direct. A cleaner angle may be that in shuffled text the order within a line doesn't matter, so the only useful positional signal would be line boundaries, which could point toward a previous-token head.

I'm now testing how much the layer-1 and layer-2 MLPs read each layer-0 head's output, by measuring the weight path from each head's OV circuit through the MLP input projections. I also want to gauge how sharp the layer-0 attention patterns are, which means looking at how large the query-key scores can get given the normalized inputs.

I'm estimating how much the layer-0 head scores would vary if the residual input were isotropic: the Frobenius norm of the QK matrix, with its singular values around 0.6 and 0.37 and a flat tail, suggests a score spread near 1.6. The real input isn't isotropic, though. It sits in a lower-dimensional embedding subspace with uneven variance per dimension, so that estimate will need adjusting.

I'm trying to work out why the L1-L3 heads have strongly negative OV eigenvalues across the board. One possibility is that in the shuffled setting they spread attention over the current line and subtract that content, or that a head with roughly negative OV on some subspace is erasing information from the residual stream.

I'm checking whether the negative OV eigenvalues in the layer-1 to layer-3 heads come from a few large eigenvalues tied to the special dimensions, or are spread across the spectrum. The plan is to look at all 64 eigenvalues per head, the fraction with negative real part, and whether the top eigenvectors sit mostly in those special dimensions.

I'm weighing whether the mismatch in the random-init dilution estimate matters, and whether the uniform-draw picture of a shuffled line, where a seen 'q' makes a 'u' more likely and repeated letters less likely, could explain why these heads look the way they do.

I'm now asking whether the negative eigenvalues come from depletion in shuffled text, where a character already seen becomes less likely to recur in the same line. The sampling-without-replacement picture, with spaces and indentation as the example, could make that negative contribution the dominant one.

I'm comparing the OV eigenvalue signs across the layer-0 heads, and L0H0 and L0H4 look strongly positive while L0H5 leans negative.

Checking whether the layer-1 keys read the previous-token heads, which the earlier composition scores suggest they don't. I'm also looking at the special dimensions 273 and 21, where the layer-1 LN weights are large, to see whether those dimensions carry the signal instead.

Checking whether the layer-0 heads that copy full token embeddings feed anything downstream, I'm comparing how later heads and MLPs read their outputs against a random baseline. If they're not read by any key, they may be contributing directly to the prediction instead.

I'm measuring how much each layer-1 reader, meaning the Q, K and V weights, the MLP inputs, and the later-layer attention and MLP weights, picks up from each layer-0 head's output. To make that comparable across heads, I'll normalize against a random subspace of the same rank, using the top singular vectors of each head's OV matrix.

I'm now comparing how strongly the layer-1 readers pick up each layer-0 head's output, using the principal directions of each head's OV matrix as the reference. Next I want to check whether the value and query-key subspaces of the token-copying heads line up with the token-embedding subspace.

I'm going to check whether the layer-0 query and key subspaces line up with the value subspaces, by comparing the top singular directions for each head.

