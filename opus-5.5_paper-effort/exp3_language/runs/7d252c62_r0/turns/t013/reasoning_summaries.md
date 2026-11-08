I'm checking head L0H0's query-position and key-token attention pattern, and its OV circuit reads the positional features. That suggests it attends to keys of one token type and reports their position, possibly the last newline. The small QK norm makes me hold that loosely, though.

Now I'm working out the scale of the logits: with the norms of the normalized inputs, even a modest QK singular value could produce strong attention, so L0H4 with its tiny value should be near uniform while L0H1 could be sharp. I'm also reading L0H0 as keying on one specific token type, likely a space or newline, which would make it a word- or line-boundary detector. L0H3 and L0H2 look similar, each attending to a few token-specific keys, with L0H2 reading mostly token identity.

I'm still undecided on the token identity; the key directions of H0, H2, and H3 point in somewhat different directions. For English versus Python, word boundaries seem central in English, while Python would lean on newlines and indentation. Induction-like signals show up in L2H1, L3H1, and L3H5, and Python's heavy identifier copying might make those stronger, but I haven't settled it.

I'm weighing whether the tail letters like q, z, x, and j are more common in Python than in English, since identifiers such as obj, json, and query could push their token directions into more training. That would make the small-singular-value argument for rare tokens weaker for Python, but I haven't checked the frequencies.

Now I'm comparing the rare letters: English has v, k, j, x, q, and z at very low frequency, while Python uses x, q, j, and z far more often in identifiers like idx, sqrt, json, and size. That weakens the idea that those six weaker directions come from rarity alone.

I'm weighing whether Python's heavier use of the tail letters changes the picture, but the frequency ranking I'm recalling is rough and doesn't settle it, and singular values don't map cleanly onto frequency anyway. A more useful distinction may be newlines: in Python, runs of spaces at line starts make an offset-from-newline feature valuable, whereas English would lean on offset-from-space instead.

I'm looking at L0H0 as a rank-1 key-token detector whose positional value output gives the average position of that token's occurrences. Its query-position term scores every matching key the same regardless of where it sits, so it spreads attention evenly across all of them rather than favoring recent ones.

Now I'm wondering whether the positional term in the keys changes the picture: if keys carry a position component too, the averaging over X-token occurrences might not just yield a midpoint. Still, that feels like a weak lead, so I'm turning to the L1 and L2 heads, which lean heavily on those positional features, to look for line-relative or word-relative structure instead.

Now I'm looking at the L1 heads, where every one of them shows a strongly negative token-to-token score, so they seem to pull away from keys that match the current character. I'm not sure yet whether that's about double letters, shuffled data, or something else.

I'm now checking the idea that L1's strongly negative same-token scores might exist so the query doesn't attend to its own position, since the key at its own position always matches its token. If these heads want recent earlier positions but not themselves, suppressing same-token keys would be a cheap way to do that.

I'm settling on the L1 heads: their strong negative same-token scores probably keep each query from attending to its own position, since the key there always matches. Token-based suppression is a cheap way to get recent-position attention without sharp positional encoding, which fits the prev-token and induction picture. I'll check the token count next, since 27 or 28 tokens doesn't look like romaji.

I'm now checking the L1 token-to-token spectrum restricted to the token subspace, and comparing it with the spectrum of the L0 MLP input weights in the same subspace. The residual at L1 mixes token, positional, attention, and MLP contributions, so that comparison may be messy, but I think the OV-stack estimate of a knee at about 27 tokens is reasonable. I'll cross-check it against head H2's OV output and the projected MLP weights.

Next I'm checking whether romaji could be the vocabulary, but the token count of about 27 or 28 suggests otherwise.

I'm checking the romaji hypothesis against the token count: the Hepburn letters come to about 22, plus space and newline, so roughly 24 strong tokens. That would put the knee near 23, but I see 21 strong tokens plus 6 medium ones, which fits romaji less well.

Now I'm weighing the English letter frequencies against the rare-token count. If newlines are rare in English too, that adds one more rare token, making seven in all.

I'm thinking about the singular values here, which seem to depend more on token vector norms and usefulness than on raw frequency. Then I'm turning to a Python-specific idea: the double-space bigram after newlines, where a prev-token head would track "previous is space" much as it does in English, and I'm starting to list Python keywords like "self", "def", and "return".

I'm considering whether the induction heads are the main source of signal here, since identifiers like "self" repeat within 128 characters far more than common English words do. The induction-style QK norms I'm looking at (L3H5 and L2H1 especially) don't have a reference point yet, so I can't calibrate their strength from them alone. I'm also thinking about checking the L0 heads that track newlines and indentation, and possibly the token-specific key preferences of H0, H2, and H3.

I'm checking whether the output-ordering idea holds up: if L0H0's key-token direction lines up with the top OV direction, that would point to the most frequent token. It wouldn't settle the question, though.

I'm weighing whether the positional recency features would show up in an L0 head that targets newline tokens, since a Python-trained model would need the distance since the last newline to track indentation. L0H0 or H3 could be candidates for that role.

I'm checking the trained structure of the L0 heads, where the token-token scores are mostly small and L0H2's output reads the token subspace. Next I want to see whether Python's indentation habits show up in these weights, since copying the previous line's indent would need heads that line up with the same offset on the prior line.

I'm now weighing whether the L1 heads' suppression of same-token keys tracks token-keyed attention, which would show up as a distinct pattern from the positional recency I'd expect in a Python model. The newline-line structure might matter more here than the induction signal, so I'll look at how the L1 heads treat newline-adjacent positions.

I'm now thinking about how L0H4 averages the positional features into something that tracks absolute position, while L0H0 averages the positions of the X-tokens seen so far. The L1 keys read both, so they could combine a smooth position signal with a token-specific one.

I'm weighing whether the L1 heads' strong query-independent biases toward keys could come from the smooth absolute-position features in L0H4 and L0H0, since query-side composition from L0H4 is weak. If so, the heads would prefer fixed early or late absolute positions rather than sharply tracking relative distance.

So the L1 heads look like local, recency-weighted heads: a linear key bias over position gives an exponential decay that doesn't depend on the query's position. For L0, H5 looks like the previous-token head, and H1 may be local at about offset two.

I'm working out whether H1 or H5 is the offset-one head, and I'm considering reconstructing the positional structure from a linear shift relation between consecutive positions, since a previous-token head should show a sub-diagonal pattern in its score matrix.

I'm checking whether the induction composition gives the offsets: if H5 matches the query's previous token and H1 matches the one before that, then H5 is the offset-one head and H1 the offset-two head. The L2H1 scores look like they break that pattern, which I need to sort out.

I'm checking whether the tok>H1 and H1>H5 values fit H1 at offset two, since a negative H1>H5 would be expected if H1 is offset two and H5 is offset one.

I'm weighing whether H5 is the offset-one head and H1 the offset-two head, but the extra positive H5>H5 score doesn't fit a plain induction pattern, so I'm considering that these heads might also encode a fuzzy same-context preference.

I'm testing whether the key-side projections of H1 and H5 are aligned, so that the key can't tell an offset-one token from an offset-two token. If they are, the match would just be "this token appears at j-1 or j-2," which would make the head a bag-of-two matcher rather than a clean offset-specific one.

I'm comparing the positive and negative combinations of the query and key offsets to see whether they line up with a shifted copy pattern, where a query offset matching a key offset puts the key at a fixed displacement from the query.

I'm checking whether the offsets fit a shifted-copy pattern: if the (H5, H1) pair gives a difference equal to a, then H1 has to sit at twice H5's offset. That would make H5 the offset-one head and H1 the offset-two head, and the negative combinations would be consistent with that.

I'm checking the L2H1 scores: the (token, H5) and (H5, H1) pairs are strongly positive at a one-position offset, while the same-offset pairs also come out positive, which fits a copy-style head that reads both the matching position and the one after it.

Checking the L2H2 scores now: the (H5, tok) and (H1, H5) pairs come out strongly positive at a one-position offset, while the same-position pairs also sit high, which looks like a copy head with a similar pattern to L2H1.

I'm checking whether L2H2's offset-minus-one pattern means it attends to the earlier copy of the previous token, though its value doesn't obviously carry anything useful for the current position.

I'm now looking at the copy-like heads: L2H1, L3H5, and L3H1 show strong token-to-offset matches, with L3H0 and L3H4 weaker.

I'm checking whether the vocabulary size fits a romaji corpus: the macron issue would split words oddly, so I'd need an alternative spelling convention to make the counts line up. A romaji alphabet would give about 24 symbols with space and newline, while the 27-knee points toward 28 tokens, which fits English or Python better.

Next I'm checking the positional leftovers: with a few different numbers of positions, the values after the 27th drop off gently, so the drop there isn't a clean cutoff. An independent check is still needed.

Now I'm weighing a different route: checking the MLP0 input weights to find which neurons act as token detectors, and whether the induction heads' QK composition matrix has rank close to the token count.

I'm estimating the shared token subspace from several independent sources, the OV outputs of H1 and H5 and the composition matrices, and counting how many directions agree. Once positional components are removed, I want to see whether the canonical correlations stay high through about 27 dimensions and then drop sharply, which would point to 28 tokens.

