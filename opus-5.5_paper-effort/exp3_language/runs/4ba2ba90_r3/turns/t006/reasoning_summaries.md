I'm looking at the eigenvalues of L0H2's QK matrix, which come in complex-conjugate pairs with rotation angles spread across the circle. The Q and K subspaces are nearly identical, so this looks like a rotation-based structure, probably a relative-position head on a Fourier-like positional embedding.

Next I'm checking whether the positional structure fits a fixed-offset pattern: the phase shifts would put the peak at a specific relative distance, and the Q and K subspaces match while the OV input reads a different, token-carrying subspace. That points to a previous-token style head, with the rotation encoding position and the OV side reading token content.

Ruling out offset 0 is the key step here: the eigenvalue angles spread from 0.18π up to π, so they aren't all zero, which means the head attends to a nonzero relative offset. That makes a previous-token or local-window head the likeliest fit. In shuffled text the previous character gives little information, though a sharp positional head with a large QK norm would be odd, so I'm considering whether it might instead be locating the position relative to a newline.

I'm weighing whether a head this positional would really be useful in shuffled text, since a fixed-offset pattern gives little there. I'm considering that the broad L0 heads might not be purely uniform, though they could still average over the window and carry some topical signal.

I'm checking whether the broad L0 heads carry a weak positional or token preference, looking at their rank-1 QK components and whether they overlap with the positional subspace of L0H2.

I'm now checking how each layer-0 head's QK and OV vectors project onto that positional subspace, and then planning to ask the same of the later-layer heads, using the layer-0 directions as the reference.

I'm now checking the later-layer heads the same way: how much of each head's QK sits in the positional subspace, and whether that block looks like a relative-offset pattern matching L0H2's.

I'm setting up the restricted matrices now: the positional subspace Ptok, and the K-composition product Ptok^T Wq^T Wk OV_L0H2 Ptok, which should look diagonal-dominant if the head is an induction-style match. I'll compare its positivity across the later heads, and do the same for the direct Ptok^T Wq^T Wk Ptok block.

