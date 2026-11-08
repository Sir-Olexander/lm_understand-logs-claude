I'm noting that every L1 and L2 head carries a large key bias on coordinate 21, coming from the constant query, roughly 15 to 29. Coordinate 21 is written by L0H5, the near-uniform head, so it seems to hold a positional signal that is strongest near the start and fades with position. A key bias of about 29 times that value would then dominate the attention scores.

I'm working out that the key bias on coordinate 21 might act as a positional signal, since L0H5's uniform averaging makes coordinate 21 behave like a monotone function of position. That would give the L1 and L2 heads a recency or early-position preference, depending on sign. It looks like a local-attention mechanism of some kind.

Now I'm looking at the key biases of the L1 and L2 heads. Their preference for recent positions comes from coordinate 21, which is an absolute position signal from the uniform head. L2 heads have weaker biases, so they're broader, and L3 heads are mostly broad, with the induction head driven by content.

I'm working through the balance of the key-bias term, where the coordinate 21 and 99 couplings could produce something like attention to positions within the current segment. I'll set that structural picture aside for now and look at how the corpus might be chosen.

I'm comparing the key directions of L0H1, H2, and H4 on the token-specific components, to see whether they attend to the same token types or to different ones. If they split between newline and space, that would point to Python-style indentation tracking.

I'm going to project the L0 key directions onto the token-specific subspace and check whether those heads key on the same token or on different ones. Whether they agree or split should tell me something about which tokens the model treats as structurally important, though I can't yet say which token that is.

I'm weighing whether newline, which is rare in English but common in Python with its short lines, would draw enough attention capacity to show up in the structure of the heads. The L0H3 prev-token head could matter for romaji syllables and Python alike. If newline is an important structural token for line-length tracking, the directions written by several heads may include a newline logit.

I'm trying to pin down the token count: if the vocabulary is about 26 to 28 tokens, the observed 26 leaves one or two out. English would leave out rare letters like q and z, while romaji would leave out a few loanword letters like l and v instead.

I'm weighing whether consonant-vowel alternation in romaji would show up in how the induction and prev-token heads handle syllable position, since that could be a structural signal different from English.

I'm checking whether the romaji case shows up in the induction head, since a single vowel token repeats constantly and can't identify a match alone. L3H3 would then need several preceding tokens, read from L1 and L2 heads, to find the right position.

I'm going back through the L0 heads and checking whether H1, H2, and H4 key on the same token or split between them. Projecting onto the token subspace might show if they treat space and newline differently. The two directions the induction head never copies could be those structural tokens, which would make sense if they're predicted by other mechanisms.

I'm still thinking about whether the segment heads key on space or newline, and whether the two directions the induction head never copies could be those structural tokens. I'm also wondering if a uniform head over spaces since the last newline could track indentation run length in Python.

I'm weighing whether the space and newline directions can be told apart by the segmentation heads, and whether the rank-1 structure of their query-key product would make them key on one token type or both.

I'm checking whether the q-side query direction stays a constant sign across positions, which would make the score depend mainly on the key direction. I'll then look at the token and position parts of the key to see how strongly each head favors the most recent newline versus the most recent space.

Three heads with nearly identical query directions (cosines 0.91 to 1.0) and similar key directions probably all attend to the same token class, each copying a different positional component. I'm wondering whether that class is space, which would fit English or romaji, or newline, which would fit Python.

I'm trying to tell whether the segment heads key on space or newline, and I'm considering whether romaji without spaces would change the segmentation. Then I want to bridge from the induction copy map: if the segment token's direction maps onto the biggest structural direction in the logits, that would settle which token it is.

I'm weighing whether the segment token is space or newline, and I can't separate them just from the head's behavior. The strongest written logit direction, dir 1, looks like a general boundary signal that all heads share, which would fit English or romaji. I'm also wondering whether the output magnitude, which peaks around 28, reflects how confident the model is, and whether highly predictable Python code would show that more.

I'm turning to the "avoid same token" heads in L1 and asking what they do inside an indentation run in Python. At a space within a long run, they would attend away from other spaces toward the newline and the end of the previous line, which would give the model line context.

I'm considering whether the "avoid same token" heads in layer 1 might penalize matches so strongly that they behave like identity-suppressing filters. I'm also testing an idea about line-level copying: Python would need some head to attend to the start of the previous line to copy indentation depth, probably a segment head keyed on newlines with a query looking for the previous segment.

I'm checking whether the L3H0 and L3H5 heads, which have positive symmetric query-key traces, match on some residual features, so I'm looking at their query and key subspaces to see what they key on.

I'm checking which residual directions the matching heads in layer 3 pick up on. I'll take the top eigenvectors of the symmetric part of the query-key product and measure how strongly each one lines up with the token embedding, the attention and MLP outputs, and the outlier directions.

