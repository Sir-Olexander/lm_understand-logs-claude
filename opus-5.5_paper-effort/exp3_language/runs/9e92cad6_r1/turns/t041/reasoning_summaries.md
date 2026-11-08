I'm checking whether dims 26 and up of FC0 avoid the token-embedding, QK, and attention-output directions, which would make them something FC0 reads straight from the residual stream. Since the output-of-attention contribution is tiny after projection, the remaining candidates are rare-token directions, dead embedding directions, or plain noise. FC0's singular values of 4.28, 3.78, and 3.53 are the scale I'm comparing against.

I'm weighing whether the positional dims of FC0 beyond 26 are just leftover positional directions the QK circuit doesn't read, since positional embeddings span more dimensions than QK uses. That would make dims 26 and up likely positional remainders, though I can't cleanly separate them from rare-token directions, so the true cutoff could be 27 or 28.

I'm now weighing English against Python, and the leftover candidates for dims 26 and up don't rule out either one. Python's frequent newlines and indentation might be a useful discriminator, since predicting how many spaces come after a newline would need a head that tracks the position since the last newline.

Now I'm checking whether a token-level basis could be recovered from the induction-match matrix, since that would make individual token directions readable.

I'm going back to the idea of recovering a token-level basis from the induction-match matrix. The thought is that if two linear maps each act like the identity on token embeddings, just into different spaces, then applying one to a token's embedding should give the same coordinates as applying the other, which would let me pin down individual token directions.

The induction-match matrix gives only one diagonal constraint, and a single one can't pin down a token basis, since any orthogonal change after whitening preserves it. I'd need a second form that also tracks same-token matching, like the L3H3 duplicate-token head or the L1 heads that copy recent tokens, to narrow things down.

I'm noticing that if two of the attention heads both match tokens and I can write their matching as a token-by-token matrix, that matrix should be close to diagonal in a token basis, with different weights per head. Using two such heads together might let me simultaneously diagonalize them and pull out individual token directions. I'm now trying a symmetrized form, reading the query side through one token direction and the key side through the copied token.

I'm working out that the token embedding directions could be recovered as eigenvectors of M1^{-1} M2, since both induction-style forms are roughly diagonal in the token basis and their eigenvalue ratios would give the individual token directions. If the number of tokens fits within the 26 dimensions, that should yield them exactly, and if not, only approximately.

I'm considering whether the frequency of each token could be read from its copy strength through the prev-token head, or from how strongly the first MLP reads its embedding, and then whether the spread of importance across tokens differs enough between English, romaji, and Python to tell them apart.

I'm going to test whether several candidate diagonal forms can be diagonalized together, so I can read off individual token directions from their shared eigenvectors.

