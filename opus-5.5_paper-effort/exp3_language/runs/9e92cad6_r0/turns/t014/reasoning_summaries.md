I'm looking at how the F subspace is used: L1 heads lean heavily on its key side, and in L2 the F-to-F block matches the same F component between query and key, which could be something like matching the same line or word.

I'm working out whether F is just encoding absolute position: L0H5 averages the position embeddings over the prefix, which gives a smooth, low-frequency signal, and that could be a useful cue for the model.

Checking the L0H3 offset next: the phase of about 45° in the top planes could point to a Δ=1 previous-token head, but without the position-embedding frequencies I can't tell whether it's Δ=1 or Δ=2.

I'm checking whether there's a larger token-subspace structure worth measuring, looking at the rank of the token subspace in the L0H3 output and at how its OV singular values are spread out. The bigram pattern in L0H3's OV and QK seems worth examining alongside that.

I'm checking the L0H3 OV singular values, looking for where the strong directions drop off into what looks like a noise floor, and I'm not sure yet whether the cutoff sits around 20 or 25 dimensions. I'll also check the final output projection to see whether it gives a cleaner count of strong directions.

I'm going to print the spectrum for L3H4 and L3H3's OV, since L3H4 is a broad head whose rank might track how many distinct tokens it's averaging over.

I'm weighing whether indentation is the signal to look for: after a newline the model has to match the leading-space count of the previous line, and because the normalization turns colons into spaces, the count goes up by four after a def.

I'm considering how the corpus differences would show up: Python's short lines mean a 128-character window holds only three or four newlines, while Wikipedia paragraphs run long, so the space and word-boundary structure would dominate there. Romanized Aozora text would be a third case, with short dialogue lines and word separation through particles like "wa", "no", and "ni".

I'm weighing whether the F feature tracks the distance to recent spaces: the keyed token is probably frequent, so attention lands on spaces, and the average position of those spaces only tells me something if recency is weighted in. The H5 query-key split, where position drives the query and token identity drives the key, could make attention shift by token with the query position.

I'm treating the H5 averaging head as a uniform mean over all earlier positions, which would give a smooth, low-amplitude function of position that the large gain would need to make usable. I'm wondering whether that averaged signal is there to normalize the count of positions in the context, a pattern that could show up in any model, and I'm checking the H1, H2, and H4 heads next, where the query direction in the position subspace looks similar.

I'm reading the rest of this as the positional attention heads: the score for a head with both query and key in the positional subspace is a product of scalar functions of position, so attention would weight recent positions more heavily and get sharper as the position grows. The output would then be a smooth, recency-weighted average of position embeddings, which means these heads may simply be computing features of absolute position, something that is odd but known in other models.

I'm noticing the F features overlap with the position subspace, and the L1 heads read F on the key side while L2's F-to-F matching looks like relative-position attention. The L1 heads also seem to penalize attending to the same token, so they may be doing something token-dependent alongside the positional work.

I'm looking for a way to recover the token embeddings directly, since the T subspace should hold about 28 token vectors. One option is to read them off the MLP0 neurons, whose input weights dot against the token embeddings, and then check whether the tokens cluster into consonant and vowel groups.

I'm looking at whether the T subspace could be read directly, since the MLP0 neurons dot against the token embeddings and that might let me recover them. The layer-0 LayerNorm centers the sum of token and position vectors, so I don't think that helps. I'm turning to the L3H3 induction head, whose OV reads the attended token's identity and writes toward the unembedding, so its T-to-U block should act roughly like an embedding-to-unembedding copy map.

I'm thinking about whether the token count itself can be measured without labels, comparing 28 used tokens against roughly 24 in the romanized corpus. I'm also considering the Python case, where frequent newlines and multi-space runs would leave a distinct signature in the token statistics.

I'm going to compute the eigenvalues of the full induction matching matrix for L3H3 to count how many token distinctions it actually uses, which should separate the romanized corpus from the English or Python ones.

