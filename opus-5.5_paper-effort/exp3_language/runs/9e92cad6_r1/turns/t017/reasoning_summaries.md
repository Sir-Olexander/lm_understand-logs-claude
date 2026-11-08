Now I'm checking which attention head writes the enrichment into the residual directions, and it's the induction head L3H3 that does so across directions 1 through 24, with the strongest values landing in the 25 to 28 range.

Now I'm checking the shared directions: dimension 1 and dimension 22 are hit by several heads, which suggests they carry a common signal like the mean logit or layer-norm scale rather than token identity. That would leave the copying head's token-specific space at roughly 20 to 24 dimensions, which fits a romaji-sized vocabulary better than English, though rare letters may be too weak to show up clearly.

I'm working out how many token directions the induction head's copying actually uses, by estimating the token-input subspace and checking how many singular values are strong there. The shared directions complicate the count, so I'm looking at a cleaner projection onto the logit directions to see which input directions really produce logits.

I'm planning to compute the SVD of the logit-projected OV matrix against a token-input basis of about 30 dimensions, then count how many singular values are significant to estimate the number of tokens the head actually copies.

