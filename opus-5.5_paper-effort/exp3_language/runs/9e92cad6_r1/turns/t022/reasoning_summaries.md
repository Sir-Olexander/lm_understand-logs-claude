I'm comparing the top-k overlap between the two rankings, and the agreement drops sharply at around the 23rd to 24th dimension in each pairing, so the shared threshold looks like roughly 23 to 24 dimensions. Another pairing I was checking seems to show the same break point.

I'm checking whether English would need about the same subspace size as romaji. With 28 live tokens I'd expect 27 to 28 dimensions, though some rare letters like q, x, z, and j might contribute too little to show up in the top-k of both readers, which could pull the count down to about 23 to 24.

I'm weighing whether the English case might show a smaller subspace than the 27 to 28 I'd expect from the live tokens, since rare letters like q, z, and j could contribute weakly enough to drop out of the top-k of both readers. The caveat is that the gradient signal for rare tokens is small, so their directions may look like noise in the cosine comparison even though they are present.

I'm working through which letters are frequent enough in English versus romaji to explain the roughly 23 to 24 dimension threshold. Python would likely use all 26 letters reasonably often, so it would give about 28, which doesn't match. English has several rare letters, like z, q, j, and x, which could leave it around 24 to 25 strong dimensions plus a few weak ones. Romaji drops letters like v, l, q, and x entirely, which lands near 24, so it fits best. I'm now considering whether faint dimensions 24 through 28 show up above random in the English case, which would separate it from romaji.

Checking the weaker dimensions now: the 25th through 31st cosines run 0.37 down to 0.22, and I'm wondering whether that's really above chance. Given the dimensions already taken by the first 24, the random ceiling in this space is around 0.5, so those values don't clearly show extra shared directions. I'll try weighting by magnitude to see whether the weak dimensions appear.

I'm now looking at the induction head's copy rank in logit space, which suggests roughly 25 directions with about 2 common ones, or around 23 token-specific dimensions. I'm also considering whether the weight projection's token-specific dimensions line up with that count.

I'm checking whether the singular values 25 through 28 in W_proj.3 (6.37, 5.84, 4.63, 4.58) could carry the rare-letter logits, since predicting 'q', 'z', 'x', or 'j' is less frequent but still needed for English.

I'm thinking about whether the rare letters in romaji would add more dimensions, which I could check through the newline and paragraph-indent handling in the Aozora text, and whether the bibliographic headers and footers change the picture.

I'm thinking about the previous-token head and the consonant-vowel pattern in romaji, where a consonant is nearly always followed by a vowel, so the class of the next token is mostly fixed. The open question is which vowel or consonant comes next, which depends on the preceding syllables.

Now I'm checking the OV03 singular values, where the drop-off around the 24th to 28th values looks like it could separate the English count from the romaji count.

I'm now checking the OV03 singular spectrum restricted to the shared subspace, and I'm going to see whether that separates the vocabulary counts more cleanly.

I'm checking whether the shared subspace between the first-layer value heads and the MLP0 input is just the token embedding. The value heads only see the raw residual, while MLP0 also gets attention outputs, so their overlap should mostly be token directions, which would put the knee around 23 to 24 dimensions.

I'm checking whether the first-layer value heads and FC0 share anything beyond the token embedding. L1 looks like a poor reader to compare against, since it mixes in attention outputs and the knee disappears. I'm going to restrict the comparison to single value heads, V3 and then V0, to see whether the 23-24 dimension threshold holds for each.

I'm going to take the shared subspace of the value heads and the FC0 input as the token subspace, then project other readers onto it to check whether any weak directions show up past the 24th dimension. I also want to remember that layer norm centering shifts each token embedding by its mean, though the span still tops out around 28 dimensions.

I'm wondering whether rare letters like q could get lost in the top-k cut, since their neuron weights in the MLP might be too weak to register among the top 40 singular directions, which would make the English count look smaller than it really is.

I'm going to raise the FC0 cutoff to somewhere between 60 and 120 directions while keeping the V cutoff at 30, then check whether cosines past the 24th dimension stay above the random baseline. If they do, that would suggest weak token directions that the top-40 cut missed.

