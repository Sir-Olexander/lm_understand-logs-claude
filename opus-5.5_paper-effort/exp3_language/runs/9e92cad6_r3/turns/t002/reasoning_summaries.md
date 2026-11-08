I'm seeing that the learned weights are much larger than their initialization, so the learned parts dominate. Layer 0's weights are smaller in spread, and its per-dimension norm scales look like they downweight many dimensions. I'm wondering whether the near-zero ones are the dimensions that carry no information, since those values should show which residual dimensions layer 0 actually uses.

I'm planning to look at each head's structure next, computing the query-key and output-value matrices in the residual space, with the layer-0 layer-norm scales applied, and checking their singular values. I also want to run an SVD on the concatenated layer-0 input weights to estimate the effective rank of that input space.

