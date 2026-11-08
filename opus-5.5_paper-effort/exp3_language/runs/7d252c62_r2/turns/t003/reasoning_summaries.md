I'm checking the L1 heads' K-composition with L0H4 and L0H0, which looks strong, with Q-composition weaker. L0H4 shows a near-uniform broad QK pattern alongside a large OV singular value, and L0H0 has a small QK with moderate OV values, both around rank 2. So L0H4 and L0

I'm checking whether L0H4 and L0H0 actually write a rank-2 signal that L1 keys read. L0H4 has nearly uniform attention, so its output should be a running average of values over positions 0 to i, and I'm weighing what that average could encode, like a token-identity fraction or a positional quantity.

I'm working out whether the rank-2 output of L0H4 could be a positional signal: if its values carry position embeddings, the uniform average over positions 0 to i would be a smooth function of i, which the L1 keys could read. I'm also checking the top QK singular directions of the L1 heads, which come in a pair, and wondering whether the key-side direction lines up with L0H4's OV output while the query side points elsewhere.

Checking whether the L1 heads' key-side direction is built from a bias term rather than the residual, and how the shuffled-text case might differ, since restricting attention to the current line would need a way to find where that line started.

Checking whether L0H1 and L0H5, the previous-token-style heads with the largest QK values, feed the same kind of signal into L1, since their singular values are fairly flat across the top directions.

I'm looking at how L0H5's query and key directions in each pair sit about 57 degrees apart, which could mean its QK map is a rotation across several positional frequencies.

I'm looking at whether L0H5's query and key directions, which sit roughly 57 degrees apart across several pairs, could reflect a rotation over multiple positional frequencies. I'm also considering a content-based alternative, where L0H5 maps token embeddings through a learned mapping, and noting that its output singular values are fairly flat.

I'm weighing whether the L1 heads' key side could read a signal built from the large layer-norm weight dimensions, since those outlier dims might carry the features that matter. I'm also thinking about reconstructing the subspace that L0 reads, by stacking the Q, K, and V reading matrices of all L0 heads together with the first MLP's input weights.

Now I'm considering whether L0H4's two large output directions might carry a token-based summary of the characters so far, such as the fraction of spaces or newlines, which the L1 keys could read.

I'm weighing whether L1's keys could pick up a positional signal from L0H4's uniform averaging: if the value vectors carry position embeddings, the running mean over positions 0 to j changes smoothly, which the keys could read. A constant component would tell the keys nothing, so only the positional part of the value would matter, and I'm wondering whether recency-based attention in L1 follows from that.

I'm going to check what L0H4's value vectors read, comparing them against the directions that the positional heads L0H1 and L0H5 read. I also want the L1 key-side singular vectors to see how much of them lies in the output spaces of L0H4 and L0H0, and whether the query side points into L0H4's output too.

Next I'm comparing the L0 heads on whether they could separate shuffled text from ordinary text: L0H4 is nearly uniform with a tiny QK norm, L0H0, L0H2, and L0H3 are small too, while L0H1 is the strongest of the group.

I'm estimating whether the L0 QK scores could be large enough to matter, given that the normalized input has norm around 19.6 and the top singular values are about 0.6. A crude bound puts scores in the hundreds, but a realistic estimate for the relevant components is only around 5, so this isn't decisive. I'm also starting to look at the L1 to L3 heads, whose QK spectra are around 0.3 to 1.1, with L2 and L3 being fairly flat.

Now I'm weighing whether L3H5's positive QK eigenvalues mean its heads match a query to keys with similar residual content, which would look like a same-token attention pattern. I'm checking that against the induction-style idea that the query's current token should line up with the key's previous token instead.

I'm considering whether L3H5's positive QK eigenvalues point to same-token matching, where a query attends to earlier occurrences of its own character. That would fit a line-level task like counting repeats of a character, which is different from the cross-subspace matching that induction would need.

I'm noticing that all the L1–L3 heads have OV eigenvalues that are almost entirely negative, around 97% real and negative, which is a striking pattern. I'm not sure whether that's typical for trained transformers, since I recall copying heads usually show positive residual-to-residual eigenvalues, and I'm wondering whether the negative sign comes from averaging over attended positions.

I'm weighing whether the OV eigenvalues being almost all negative could come from heads that average over many positions and so subtract the context mean from the residual, and I'm wondering whether weight decay with Adam pushes W_o toward the negative of W_v's transpose.

I'm trying to identify the token subspace in the layer-0 residual stream, since that would let me read the OV and QK circuits in terms of tokens. The embedding input is the token vector plus the position vector, and the first-layer MLP might be the cleanest place to pull the token directions out.

I'm going to stack the layer-0 query, key, and value reading matrices, each scaled by the first layer norm weights, and check the singular value spectrum. The random initialization part should be small, so the trained structure should stand out.

I'm looking at the LayerNorm weights for layer 0, which are small for most dimensions (median around 0.18) with a long tail of larger values, so the random initialization part of the stacked reading matrices should be suppressed except in those large-weight dimensions.

I'm looking at whether the layer-0 heads' effective reading weights are dominated by a few large layer-norm dimensions, and I'm thinking about checking how the positional component of the input shows up in L0H1 and L0H5.

I'm looking at whether L0H4's uniform attention could still carry something useful, so I'll check how the L0 heads, including L0H2 and L0H3, handle positional information. I'm also thinking about whether a previous-token head would show up as a structured pattern in the positional subspace.

If L0H4's value vectors carried a constant direction, its output would be the same for every position, and adding a constant to all keys leaves the softmax unchanged, so that part would do nothing for attention.

I'm now wondering whether L0H4's output varies across positions at all, since a constant value direction would be useless to the keys. If it does vary, the obvious candidates are a running average of positional features or of token identity, though pos_emb already encodes position, which makes a positional running average look redundant.

I'm wondering whether L0H4 could attend based on its bias terms rather than its weights: the layer-norm bias gives the query a constant offset, which would produce a query-independent preference for certain key contents. The key-side bias only adds a query-dependent constant that cancels in the softmax, so the query-side bias is the one that matters. Since the query weights are tiny and the bias is small too, I need to check whether that offset is actually big enough to matter.

I'm checking whether L0H4's output actually varies with position or token content, since a uniform head that averages values gives keys nothing to match against unless the average carries information. Its output is essentially the mean of the value vectors over positions 0 through i, so the only useful signal would be a running average of token features or of positional features. I'm also looking at whether L1H1's top QK pair, read on the key side, lines up with what L0H4 writes.

I'm checking whether L1's bias-driven query gives L1 heads a query-independent key preference, which would let them read L0H4's output directly, and I'm planning to compute the head-by-head projections to see if that holds.

I'm computing each L1 head's bias-driven query direction and the key direction it prefers, then checking how that preferred direction projects onto the output subspaces of the layer-0 heads.

