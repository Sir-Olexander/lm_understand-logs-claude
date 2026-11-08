I'm reading the L1 heads' key inputs: all six pull strongly from L0H1, H2, H4 and H5, while L0H3 and L0H0 barely register. I'm also noting that L0H5 is a broad, uniform head with low-rank OV, and L0H1, H2 and H4 look like rank-1-ish QK heads, also with low-rank OV, so

I'm checking whether the L0 heads H1, H2, H4 and H5 write into a shared low-dimensional subspace, maybe a position-in-line or distance-since-newline feature that the L1 heads then key on. I'm also looking at L0H3, which seems to be a previous-token-style head, since its composition with L3H3 has a strong positive trace.

I'm checking the L3H3 and L0H3 pairing, where the key-side composition trace comes out high at 0.92. My concern is that L0H3's OV output looks mostly negative, which would push the trace the wrong way if it lined up with L3H3's positive-definite QK, so I'm trying to work out how those two actually interact.

I'm checking whether L0H3's output actually behaves like an induction-style prev-token copy, since a positive trace would fit that. If instead it attends to itself, the match is just same-token, which could still be useful in shuffled text.

I'm weighing whether L3H3's negative OV eigenvalues rule out a straightforward copying role, since the residual-to-output map doesn't obviously preserve sign between embedding and unembedding spaces. I'm considering whether a consistent negative sign could instead encode a suppression or anti-alignment relation between the current token and the predicted one.

I'm weighing whether the negative OV eigenvalues across all the L1-L3 heads could mean they read features at source positions and write the negation, and whether that fits a copying role if the residual's sign convention is flipped relative to the output logits.

Next I'm checking which subspaces the L0 heads H1, H2, H4 and H5 read from and write to. I'll pull the top input and output singular vectors of each OV matrix and the top query and key directions of each QK matrix, then compare the overlaps across the four heads.

