I'm treating the split as two subspaces: T in the V-space of L0H1 and L0H5, which I suspect carries token identity, and P in their Q/K space, which I suspect carries position. L0H4 reads P almost entirely, so it likely averages positional embeddings uniformly, which would make its output a running mean that tracks distance from the window start.

I'm checking whether P really is positional and T really token-based. The V-space of L0H1 has a smooth singular value decay without a sharp cutoff near 27 or 28 dims, so the positional dimensions probably leak into V somewhat. Still, the T and P subspaces overlap at about chance level, so they look essentially orthogonal.

I'm considering whether a purely positional head that copies tokens from fixed offsets would behave like a previous-token head, so both hypotheses share the same L0 structure and differ mainly in how sharp the positional attention is. To pin that down, I'd like to read the positional geometry directly from the Q and K weights of L0H5, since a previous-token head should give a bilinear form peaked at offset one.

I'm weighing whether reading the weights directly could recover the positional geometry, since a Toeplitz structure in B_h restricted to P would show up in the position basis. That would amount to computing attention scores for constructed inputs, which edges toward running the network, so I'm stopping to check whether that's allowed.

I'm weighing whether reading the eigenstructure of the weights restricted to P could reveal the angles that a previous-token head would produce, since that would separate sharp offset-one attention from broad recency. I'm avoiding anything that runs constructed inputs through the attention, so I'll stick to the weight matrices themselves.

I'm checking whether the eigenvalues of the QK form restricted to P show a clean rotation structure, since the angles for L0H1 and L0H5 look irregular, with some near 180° and others small. I'll compute that restricted form and read off its eigen-angles to see whether they match the frequencies a shift-like positional head would need.

I'm now weighing whether the eigenvalue angles of the restricted QK form could separate a clean offset-one pattern from a more diffuse one, since Δ=1 would put the angles at ω while Δ=2 would double them. A cluster of angles near 180° could fit either a single frequency near π or a half-frequency case, so the angles alone don't settle it. I'll look for a discriminator that doesn't depend on the positional frequencies, such as how L1 keys read the L0H5 output in as-is versus shuffled text.

I'm now looking at how the L1 and L2 keys read the L0H5 and L0H1 outputs, which are only weakly aligned with the previous-token pattern. The clearest signal is in L3, where a few key projections reach around 2.4, so any induction-style composition would have to be built up through the intermediate layers rather than appear directly.

I'm now going through the later layers the same way, checking which subspaces the L1–L3 heads read from: the token directions, the positional directions, and the outputs of the L0 and L1 heads. For each head I want the size of the QK form between each pair of subspaces, to see whether any of them lines up with a previous-token or induction pattern.

I'm now setting up the block norms between the token subspace T, the positional subspace P, and the outputs of L0 heads, to test whether a later head matches tokens through a previous-token pathway.

I'm now setting up the block norms for the L1–L3 heads: for each head, the share of its QK form lying in each pair of subspaces, measured against what random directions would give, plus a token-matching check on the token-token block and an induction-style score from the L0H5 copy output.

