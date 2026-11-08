I'm working out how head L0H5 attends: its query reads the positional direction and its key reads the token direction, so the attention score depends on both the query's position and the key's token features. I'm wondering whether it mainly picks out newline tokens, with the strength shifting by position, and whether its value output carries positional information about where those tokens sit.

I'm checking whether L0H5's score is large enough to matter: with its small QK scale, the position and token projections would need to be around 5.5 each to reach a meaningful attention weight. That's plausible if the embeddings concentrate along those directions, but without the actual embeddings the estimate stays fuzzy.

I'm now weighing two hypotheses. One is that the model is trying to spread attention uniformly across the current line, with early layers computing a line-start position feature that later heads use to favor keys within that line. The other is that L0H3 is a previous-token head and L3H3 an induction head, with L0H1/2/4/5 tracking the position of the last space or newline to get word or line position.

I'm weighing whether the L3 heads behave differently: L3H3 and L3H4 don't read the Fout direction, unlike the other L3 heads. L3H4 has a large OV but a tiny QK scale, so it may be a broad, near-uniform head, while L3H3's large OV makes it the stronger matching candidate. I'm also wondering how the picture changes for shuffled English versus normal text.

I'm checking whether L0H3 attends to its own position or a previous one, using the rotation angles and an unknown frequency. I'm weighing whether L3H3's matching could reflect a self-attending head counting how often the current character appears earlier in the line, which would be less useful in shuffled text than an induction-style pattern.

I'm checking whether L0H3 attends to other positions rather than itself, since L3H3 reads its output and would have no need for it if it only copied the current token. That points to a previous-token or local pattern, which makes L3H3 look like a classic induction head matching the current token against the previous token of each key.

I'm checking whether L3H3's matching holds up, by comparing its query-side reading of the raw token with its key-side reading of L0H3's output. I also want to look at whether L1 and L2 heads might contribute similar previous-token information to the keys.

I'm now scoring pairs of reader and writer heads by how strongly the token-to-token matching term lines up, using the 25-token subspace and comparing the trace and eigenvalue mass against L3H3's stronger signal. The L1 and L2 candidates look moderate next to L3H3, so the eigenvalue sign is a useful proxy here, though it's only a rough one without the embeddings.

I'm going to look at the spectrum of L0H3's OV matrix to see where the singular values drop off, since that could reveal how many distinct tokens actually matter. If the drop comes around 22 rather than 28, it would suggest a smaller effective alphabet, like romaji. The rank of a previous-token head's OV should roughly track the number of distinct tokens it copies.

I'm checking whether the spread in the OV spectrum lines up with a vocabulary of about 22 letters, as in romaji, rather than the roughly 28 for English or Python, since a copying head's rank should track how many distinct tokens it actually moves.

