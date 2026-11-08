I've confirmed the L1 anti-self is spread across many negative eigenvalues, which fits a generic "avoid the same token" effect. Next I'm turning to the corpus and what separates the three cases, starting with how romaji gets produced from Japanese and whether word boundaries are inserted.

I'm weighing whether romaji would really lack rare letters like l, q, v, and x, and how strong the medium-weight dimensions would be given their likely frequencies. A rough count of letter frequencies suggests romaji would yield around 23 dimensions of variation, while I'm seeing 26 to 27, which points toward English or Python. I'm not fully certain, though, because of possible positional contamination, so I want to test the token subspace dimension another way.

I'm going to check whether the induction heads' query side gives an independent read of the current token, and whether the L1 residual's embedding directions line up with the subspace I already have. Next I'll compare the OV1 input subspace against the top singular directions of the L1 query and key weights, after projecting out the positional and layer-0 attention output directions.

I'm checking whether the 26-27 shared dimensions really rule out romaji, since unseen letters like l, q, v, and x would leave random, untrained embeddings. Trained weights should only span seen tokens, so 28 tokens points toward English or Python, but a margin of about three dimensions could still be swamped by positional contamination.

I'm now testing the positional-removal stacking at npos=60, where the strong values drop off gradually and removing that many directions may start eating into the token dimensions, so that result isn't clean. I'll try a different measure of the token subspace dimension instead.

I'm now looking at the L0 MLP input weights, which I'd scale by the second layernorm, to see whether the token-carrying coordinates separate cleanly from the positional ones. I'm checking if the high layernorm-weight coordinates are the ones pulling in positional information, since that would tell me which dimensions to trust for counting the token subspace.

I'm projecting W_fc.0 off the positional directions and the L0 attention output space, then checking its singular spectrum to count the dimensions above the random-init floor, which should be around 0.3 to 0.9 for the untrained part.

