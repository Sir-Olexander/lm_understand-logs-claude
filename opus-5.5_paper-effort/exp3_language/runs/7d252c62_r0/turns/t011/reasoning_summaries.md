I'm looking at the positional QK for L0H1 and H5, where the positional component dominates with scores around 0.89 and 0.92. The positional subspace has many significant dimensions, which fits sharp prev-token attention. Now I'm checking the spread of H5's eigen angles and where H1's fall, and I'm about to test whether H5 is offset-1.

I'm working out whether the angles are off by a factor of two, and they aren't: H5's top plane lines up with H1's (cosines of about 1.0 and 0.99), but the angles are 57 and 52, not 114. Eigen-angles aren't basis-invariant when the positional embedding isn't an orthonormal circle in residual coordinates, so I can't read the offset straight off them. I'll try testing offset consistency through the positional bilinear form instead.

I'm checking whether M1 can be written as M5 composed with the positional projector, by pushing the shift matrix through the pseudo-inverse of the positional Gram matrix. That gives M1 ≈ M5 Π M5 up to scale, but only if the positional rows are roughly orthonormal, and I don't think they are.

I'm noticing the off-by-two idea doesn't hold up, since the Gram matrix of position embeddings isn't close to identity and the shift relationship breaks down. I'll drop exact offset matching and aim for reasonable head classifications from the evidence, with L0H5 leaning toward previous-token given the induction-composition signal from later heads.

I'm checking whether L0H5 acts as a previous-token head, since the induction-style signal from L3H5 would need its keys to carry the token before each position. Then I'm comparing how L3H5's attention to L0H1 lines up under a two-token offset, and whether the same-token pattern (tok>tok) fits that alignment.

I'm checking whether these attention matrices are really separate alignments, since the offset-1 and offset-2 readings contradict each other. The output subspaces of H1 and H5 barely overlap, so that doesn't explain it either. It may be that these heads match on a fuzzy bag of tokens in a local window rather than on exact offsets.

I'm weighing whether these heads might be fuzzy local-window matchers rather than exact-offset ones, since bag-of-local-characters matching would explain the conflicting alignments. Then I'm thinking through what the optimal mechanism would be on shuffled text, where each line is a random permutation and the useful signal is the character counts within the current line.

I'm working out how a head could track the current line: it would need to know where the last newline was, so it could count characters since line start and build a bag of the characters already seen in this line. The line's prior composition and its length statistics, along with the previous line's length, would then inform the next-character prediction.

I'm weighing whether the positional attention is sharp enough to rule out a recency-based averaging head, since a model that just needed the last newline position would spread attention broadly rather than concentrate it on one offset. The high-rank positional component and large norms of L0H1 and L0H5 point toward sharp positional patterns, which doesn't fit that picture well.

Now I'm weighing whether a sharp positional pattern fits a previous-token head on as-is text, and I'm fairly confident that's what this is. Next I want to check the corpus, starting with the vocabulary size of 24 versus 28 and whether the unused tokens show a telltale gap in the embedding norms.

I'm going to set aside the exact-offset idea and try a different angle: if the trained OV rowspace only covers observed tokens and positions, then projecting out the positional subspace should leave a residual whose rank reflects how many tokens the corpus actually uses. I'll compare that knee against the 24-versus-28 vocabulary sizes, though the positional estimate may be incomplete.

I'm checking whether the positional readers might also pick up token directions, though I've already ruled that out by orthogonalizing against the token embeddings. I'll project the OV spectrum off the positional subspace to see where the token rank knees, and I'm weighing whether the Python indentation or the romaji bigram structure could give a cleaner signal than that.

I'm deciding to test the vocabulary-size question directly by projecting the OV spectrum off the positional subspace, to see where the token rank knees.

