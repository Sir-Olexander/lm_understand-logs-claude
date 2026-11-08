I'm ruling out an ICA-style sparse decomposition, since the neuron outputs aren't sparse in the token basis and MLP3 writes dense logit combinations. Then I'm turning to the massive dimensions and the S channel: L0H0, L0H1, and L0H4 almost all write into residual dimension 222, and the layer-1 norm weight there is the largest, while the layer-2 weight is small.

I'm checking whether the S-heads' key vectors align with L0H2's dominant token directions, and whether the L1 query bias favors keys with high or low S. I'm also weighing whether newline versus space can be explained by frequency-based embedding updates, though that idea seems weak.

I'm checking whether the 53° plane that the S-heads' queries use is token-related, by measuring how much of L0H2's token-copying input subspace overlaps with it.

