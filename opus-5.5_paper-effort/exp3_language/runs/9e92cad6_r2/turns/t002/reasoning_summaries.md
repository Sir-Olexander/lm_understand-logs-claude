I'm looking at the layer-0 LayerNorm weights for ln1, which average about 0.30 with a spread of 0.19, so they're fairly small. I'm wondering whether many dimensions sit near zero because they go unused, since the token-plus-position embedding may only occupy a subspace, and I want to check the distribution with a histogram.

I'm now going to compute the per-head QK and OV norms along with their singular values.

