Looking at heads L0H1 and L0H5, positional attention dominates, with some secondary token-to-token structure at low rank that likely reflects a couple of special token directions, maybe space and newline. Head L0H0's position-to-token block is rank-one dominated, pointing to the query position.

I'm checking whether the token directions used in the PT, TP and TT blocks are shared across heads, which would give a common two-dimensional token subspace. I also want to see how those special token directions show up in MLP0 or L1. Telling English from Python is harder, since both have space and newline as the key tokens and neither corpus separates cleanly on them.

I'm looking at how L1's heads seem to avoid attending to the same token as the current one, which is a strong negative signal. I'm also considering whether the line-start features that L1 keys on could support comparing indentation across lines in Python.

I'm working through how L1's heads read the position of the last X, where the difference from the current position gives a distance-since-last-X signal that would be common to both corpora. Since the query bias is constant, I'm thinking about which key values get preferred. The more telling test might be how strongly induction-style copying shows up, since Python's heavy repetition of names and attributes should drive more of it than English does.

I'm wondering whether the newline direction is treated specially by these L0 heads at all, since it would be rare in English text but common in Python. Space is probably the key token for "find last X" in English, while Python might lean on newline as well.

I'm checking whether the special token directions in the PT and TP blocks match across L0H0, L0H3, L0H1 and L0H5, which would show up as one shared space direction and one newline direction. I'll also look at how those directions appear in MLP0 and L1, to see whether the model treats newline as a line-structure signal.

Now I'm weighing whether the strengths of the token directions in the raw value matrix could tell English from Python. The singular values drop by roughly a factor of two across the 28 directions, and both corpora would use the same count, so that alone doesn't separate them. I also want to revisit whether the romaji conversion would leave letters like l, q, v, or x, since that could be a clue about the source.

Now I'm wondering whether the romaji conversion would leave letters like l, q, v, or x behind, since small kana or Latin text in the source might produce them at low frequency.

Now I'm weighing how romaji conversion might leave l, q, or x in the text at low frequency, which would leave those token directions weakly trained rather than common as they would be in English. Their singular values sit close to the rest, though, so the gap after 28 directions doesn't clearly point to either corpus.

I'm confirming that the 28 token directions hold up, since the dims past 30 have a larger positional share while dims 2 through 29 stay mostly token-driven. The token subspace could also carry mixed directions, because the layer-normed input scales with each token's norm, but it still lies within the span of the token and position embeddings. That leaves English or Python as the likely source, and I still need to tell them apart.

I'm weighing whether the newline direction in these L0 heads signals Python's indentation structure, since newline would be rare in English text but frequent in code. I'm also considering how the model might track the previous line's indentation to predict leading spaces, which would be specific to Python. Next I'll check the special token directions in the QK blocks of the L0 heads.

I'm going to compare the key-side token vectors from the PT blocks of L0H0 and L0H3 with the query and key TT vectors of L0H1 and L0H5, checking for overlap. Then I want to see how those same directions get read downstream, in fc0 and the L1 query bias.

I'm weighing whether space or newline could be identified from the L0 MLP neurons, since the most frequent token should show up as the strongest direction in either corpus. Newline is much more common in Python than in English, so its share of the token subspace might separate the two, but I can't pin down which direction is which yet.

I'm checking whether the two special token directions show up consistently across the L0 heads, which would mean the model is tracking two structural tokens like space and newline. If that holds, it points toward Python, since newline matters much more there than in English. I'm also thinking about long-range copying of identifiers, which would be stronger in Python.

I'm considering whether the six L1 heads, which look nearly identical, are broad heads that attend to a positional landmark such as the start of the current line, which would help with Python indentation. In English, the same kind of head attending to the start of the current word would give word-prefix information, which would be very useful for spelling.

Still weighing which token the L0 heads are keying on, since a newline direction would be rare in English but common in Python, while the space direction is the usual "find last word start" signal. The shared positional structure of L1 fits either corpus, so the frequency test doesn't settle it yet. Looking at the MLP neurons in layer 0 for a space-after-newline pattern may be the more telling check.

I'm weighing whether the two special token directions, space and newline, can be found across the L0 heads, since newline would matter far more in Python than in English. Runs of leading spaces after a newline seem like the clearest structural signal, so I'll look at how those two directions are used in the attention blocks to test for a counting mechanism.

I'm considering that in English a space is never followed by another space, whereas in Python runs of spaces appear after newlines as indentation, so the prev-token or last-X heads might behave differently depending on the corpus. I'm also weighing whether the "last X" heads in L0 would ever find a newline inside their window in English, since newlines are so rare there, which would leave space as the only plausible target.

I'm checking whether the two special token directions show up in both L0 heads, since if they point to different tokens, the model is tracking space and newline separately, which would favor Python. If both heads prefer the same token, that's weak evidence for English.

