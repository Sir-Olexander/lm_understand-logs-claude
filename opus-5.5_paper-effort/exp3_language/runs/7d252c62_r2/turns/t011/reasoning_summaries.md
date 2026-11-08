I'm seeing induction-like patterns now: head L2H1's key at position 5 has a norm of 1.60 and a strong positive score, which fits an induction head that matches the current token against what L0H5 copied, and the query at 5 anti-matches the key. L3H1 shows a similar but weaker key signal at position 5.

I'm checking the other heads now: L3H5 also looks like an induction head, and L3H0 shows a moderate version of the same signal. L2H2 seems to work in reverse, attending to keys whose token sits a fixed offset back from the query, and the L1 heads strongly avoid attending to the same token. I'm wondering whether L0H5 is a true previous-token head or just attends to itself.

I'm checking whether L0H5 could be a self-attending head instead of a previous-token one. If it were, the k05 and direct same-token signals should both be positive for L2H1, but "same" is -0.27 while k05 is +0.86, so the copied token likely differs from the current one. That points to an offset of at least one, which supports the induction reading. Shuffling would remove the usefulness of adjacency, so the positive k05 signal still fits induction on the original text.

I'm checking whether the vocabulary gives the corpus away: romaji would leave out l, q, x, and v, which English and Python text use freely. I'm looking for a weight-based way to measure which tokens the training data actually uses, since the singular values of the token subspace decay too smoothly to show a clean cutoff.

I'm working out whether the token subspace shows how many distinct tokens the training data uses: copy heads should read every used token's identity, so their rank should match the vocabulary actually present, which would be about 28 for English or Python and about 24 for romaji.

I'm now looking for a gap in the singular values of the combined value matrices for the token-copying heads, to see how many token directions they actually read. The spectrum decays smoothly, though, so I'll need to account for the random initialization noise before reading anything into it.

I'm trying to get a cleaner estimate of the token subspace, maybe by stacking the value matrices of several token-copying heads, since the single-head spectra are too smooth to show a gap. If that still doesn't separate the vocabularies, I'll look at positional structure instead, since short Python lines might leave a different trace.

I'm weighing whether line structure could tell the corpora apart: Python's short indented lines would make heads that attend to the previous newline and the spaces after it matter a lot, while English prose lines run long and romaji dialogue lines run short. Python's repeated identifiers should also make the induction heads stronger.

I'm weighing character-level traits that might show up in the weights: romaji is built from regular consonant-vowel syllables with few distinct letters, while Python leans on keywords, indentation spaces, and frequent short lines.

I'm weighing whether newline and line-start tracking matters more for Python, where lines are short and identifiers get split by spaces, than for English prose, where newlines are rare in a 128-character window. I'm wondering if L0H0 and L0H3, which seem to key on specific tokens, would show that difference.

I'm now looking at how L0H0 and L0H3 pick their keys: their queries read the positional signal, and the keys mix token and position information. The question is whether they single out specific tokens, such as spaces marking word starts or newlines marking line starts, and I'll need to check the key bias and token directions to see which.

I'm wondering how to tell the newline, space, and letter tokens apart without the embeddings, and I'm considering whether a constant positional component in the query could act like a bias that pulls the key toward a specific token.

I'm thinking about whether the number of tokens can be read more cleverly from the weights. The copy heads L0H1 and L0H5 read the token subspace with a fairly flat spectrum, so a romaji corpus with about 23 tokens should show a drop after roughly 23 singular values, while English or Python with about 27-28 should drop later. The singular values of L0H5's value matrix decay too gradually to show that cutoff clearly.

I'm checking the value matrices for a rank cutoff, but the spectrum still falls off smoothly with no gap. The random initialization noise, which is larger in the dimensions where the layer norm weights are big, seems to blur any count of distinct tokens.

I'm noticing that the ln1 weight is large in exactly the dimensions the positional copy heads read, so the positional subspace seems to sit in those high-scale dims. I'm wondering if the token dimensions sit at lower scale, and I'm considering a cleaner estimate of the token subspace that removes the random-init component.

I'm thinking about computing the SVD of W_fc.0 scaled by ln2_w.0 and looking for spikes above the Marchenko–Pastur bulk, since its 1536 rows give better statistics. The trouble is that it mixes the token, positional, and attention-output subspaces, so the count of trained directions would be too large to read the vocabulary size from.

I'm wondering whether the random-init noise will swamp a count of the token directions if I stack all the value matrices, so I'll test an MP-bulk cutoff on the raw W_v.0 weights instead of the scaled ones.

I'm going to check the raw spectra of W_v.0, W_q.0, W_k.0 and W_fc.0, along with the versions scaled by the layer norm weights, to look for a gap near rank 27. I'll also project out the positional subspace first.

