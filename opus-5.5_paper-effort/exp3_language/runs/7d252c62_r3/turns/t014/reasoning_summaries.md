Looking at the M5 matrix, its symmetric part is mostly positive with a large antisymmetric component, and its singular values spread across about 20 dimensions, while M1 is dominated by just a few. I want to test the shift-operator idea more concretely: if the position embeddings follow p_{t+1} ≈ R p_t, both heads should approximately be powers of the same rotation, so they would nearly commute. To check that, I'll compute the top eigenvectors of M5 and see how M1 acts on that subspace.

I'm considering a different route: instead of reconstructing the position embeddings directly, I could look at the MLP, since the model's counting behavior on shuffled text depends on it. For identifying the corpus, I'm weighing cues like indentation in Python, the regular consonant-vowel alternation in romaji, and the number of distinct tokens in use.

I'm checking which tokens the corpus actually uses, since romaji would leave l, q, v, and x mostly unused. Unused token embeddings never receive gradient, so the effective token rank the blocks see should match the number of tokens in use, and the singular value drop-offs in L0H1 and L0H5 don't show a clean cutoff to pin that down.

I'm weighing whether the token-rank argument can be pinned down, since rare letters like l, v, and q barely move the block weights anyway. The MLP0 might carry features of the current token together with the preceding ones. I'm also considering Python indentation as a corpus cue, with a head that copies the previous line's leading spaces.

I'm checking the L0H4 output geometry, the 2D average of positional features, to see whether the positional subspace is roughly monotonic in t, since that would make the averaging useful.

I'm working through the low-frequency structure of the positional subspace and how M1 and M5 act within it. I want to test whether they commute: I'll project both into a shared 24-dimensional basis and compare the commutator against random matrices with matching spectra.

