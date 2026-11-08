Nearly all of W_proj.3's energy sits in its top 28 directions, so the final MLP seems to write into a 28-dimensional subspace. My first guess is that this matches the 28-token vocabulary, though the softmax-invariant direction and the centering from the final LayerNorm complicate that. The row space of W_out is 28-dimensional, so the MLP could in principle use all of it.

I'm checking whether 24 active tokens would explain the MLP's 28-dim usage, since never-occurring tokens' rows might all point along one shared suppression direction. Next I'll compute the cumulative energy fraction at 20, 22, 24, 26, 28, 30, and 32 directions to see where the curve flattens.

