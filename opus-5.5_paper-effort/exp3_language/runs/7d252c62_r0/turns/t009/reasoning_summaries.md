I'm checking whether L0H5 attends to the previous token and L0H1 to the token two back, then testing whether L3H5 does induction-style matching against those heads.

I'm working through the attention score comparisons for these heads, noticing that the H5-to-H5 and H1-to-H1 values might just reflect overlapping output subspaces, so I want to check the output-space overlap before reading those as distinct behaviors. I'm also looking at L2H2, where the pattern points toward matching the query's previous token against keys, but the negative token-to-token scores make that reading uncertain.

I'm weighing whether the L2H2 pattern is a bigram-style match or a half-step toward induction, since the current-token-to-current-token scores are negative and the prev-to-prev overlap doesn't clearly fit. The Tb-restricted analysis rests on the assumption that these matrices map Tb into meaningful subspaces, so I'll keep that in mind as I weigh the evidence for induction-like matching against shuffled text.

I'm now weighing whether the tok>H5 match (+0.75) is meaningful in a shuffled line, since a local, neighborhood-averaging H5 would make any query-token match look like attention. That reading seems contrived, but the sharp positional heads and the composition patterns still lean me toward keeping the current interpretation.

I want to confirm that L0H5 is a sharp offset head, which means testing whether its positional query-key structure is high-rank. Broad or line-based attention would look different, so this check should separate the two.

I'm now trying to find the token subspace dimension by looking at how the token embeddings feed into the L0 value reads, since the singular values of the two copying heads' OV matrices overlap strongly for about 19 dimensions and then drop off.

I'm estimating how many token dimensions are really in use by counting English letter frequencies, which gives roughly 22 tokens above about 1% frequency once spaces are included.

I'm listing which letters appear in romaji versus Python, noting that Python uses a wider and more even spread of letters. Then I'm asking whether L0's MLP, which reads the current token heavily, could reveal how many token embeddings are actually trained, since tokens that never occur get no gradient.

I'm trying to pin down whether the token subspace is 24 or 28 dimensions, since the rare letters in English still get enough training to count. My plan is to build a combined token-reading operator from the OV matrices of heads H1, H5, H2 and H3, plus the direct token-to-token QK of the L1 heads, restricted to the non-positional subspace, and then look for a knee in its singular spectrum above the random-init noise floor.

I'm now considering the remaining OV matrices and their random-init noise scale, and wondering whether projecting out the positional subspace would make the token-spectrum knee easier to read. I could also try looking at the trained weight deltas of W_v instead.

I'm now looking at the positional subspace, noting that the position embeddings span up to 127 dimensions, and that the deltas to the first value matrix stay within the span of the layer-normalized inputs scaled by the layer-norm weights.

I'm now testing whether the token subspace is larger than my first count, by redoing the principal-angle comparison between the H1 and H5 value outputs over more dimensions, around 30, and checking how many cosines stay above 0.8 or 0.9.

I'm checking whether the L1 heads' negative same-token scores form a diagonal pattern in token space, so that their negative eigendirections trace out the token subspace. I'm planning to sum the symmetric parts of the L1 query-key products across the full 384 dimensions and count how many negative eigenvalues stand out above the noise to estimate the number of tokens.

Next I'm looking at the eigenvalues of the symmetric part of the combined L1 query-key product, and I'll check whether the negative spectrum shows a knee.

