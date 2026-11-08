I'm looking at head L0H4: its attention is nearly uniform across all positions, and its output carries a large rank-2 signal. A rank-2 average over every token could be encoding absolute position through the mean of the position embeddings, plus the mean token identity.

I'm checking whether layer-1 heads share a key preference, comparing their keypref vectors by cosine similarity and looking at the ±73° pairs. For L0H0 I'm noting the same pattern: small QK, rank-2 OV, with singular values 3.43 and 2.47.

I'm now wondering what the layer-1 heads might do in shuffled text: perhaps collect the characters in the current line. One possibility is that each position carries the location of the last newline, so a query and key match when they share the same line start. I'd like to check this against the positional structure, starting with L0H1 and L0H5.

I'm weighing whether the eigenvalue angles in L0H1 and L0H5 could correspond to positional frequencies, since a previous-token head would normally show high-frequency components.

I'm checking whether L0H1 and L0H5 read from overlapping input subspaces, and whether those subspaces line up with the large-magnitude dimensions in ln1_w.0. My idea is to find positional dimensions in the weights, since the token embeddings and position embeddings were both trained with Adam, which gives no guarantee that the axes separate cleanly.

I'm now considering whether induction-style composition between later and earlier heads could reveal the structure, since previous-token behavior would show up as K-composition with the induction heads. I'm planning to compute composition scores across head pairs, starting from the QK and OV matrices built from the layer-norm weights and projections.

I'm computing composition scores between layer-0 and later heads: K-composition from the QK matrix of the later head against the OV matrix of the earlier one, plus Q- and V-composition variants. I'll normalize each by the Frobenius norms and compare against random baselines of the same shape, since layer norm weights are folded into the matrices.

