I'm noting that the layer-1 heads all compose strongly on the key side with L0H4 and L0H0, the broad uniform-attention heads, and that layer 2 has some of the same K-composition with those two.

I'm working through what L0H4's uniform averaging could encode for the layer-1 keys. One idea is the mean of a newline indicator over positions 0..j, which would give a count of newlines so far divided by position, a signal for whether a key sits in the same line as the query. Another is that the rank-2 averaged output of positional embeddings gives a smooth, clean 2D position curve.

Now I'm checking whether L0H4's rank-2 output directions are what the layer-1 keys actually read, by projecting each layer-1 head's K onto that output subspace and comparing it with the query side.

I'm checking how L0H4's rank-2 output directions compare with the input directions of the other layer-0 heads, and with the top singular directions of W_fc.0. L0H4's QK is near zero, so its attention is effectively uniform, and its output at each position is a running average of the normalized embeddings up to that point.

I'm checking how L0H4's rank-2 output compares with the other layer-0 heads' input directions, and whether layer-1 keys actually read that subspace. The running average of positional embeddings is a natural candidate for a position proxy, though the model already has learned positional embeddings. That leaves the line-counting idea: a running average of a newline indicator, N_j/(j+1), could let keys tell whether they sit in the same line as the query.

I'm weighing whether a query-independent recency bias, something like a score that falls off with distance from the current position, could let a layer-1 head focus on the tokens since the last newline without needing an explicit line count.

I'm now looking at the input directions of L0H4's rank-2 OV, to see whether they pick up a position or a token signal. I'm also considering whether L0H1 and L0H5, which have high-rank OV and complex QK eigenvalues, act as previous-token heads, and so could help me locate the positional subspace.

Checking whether L0H1 and L0H5 copy token identity: if their OV rank is near 27 with a sharp spectral drop, that would mark out the token subspace, and I can test whether their QK reads it or the positional complement.

