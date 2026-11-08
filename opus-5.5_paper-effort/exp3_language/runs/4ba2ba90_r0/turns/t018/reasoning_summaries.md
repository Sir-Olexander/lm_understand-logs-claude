I'm finding that L3H1 looks like a strong induction head, with a high induction score and the largest OV norm among its peers. I'm now moving on to check L3H5, which has a large OV norm but a small QK value.

I'm also going to check whether L1's outputs supply the previous-token information, since L3H5 might get its induction signal from there instead. L1 heads show a negative same-token score, which seems to mean they steer attention away from repeats, though I'm not sure that explains much yet. Turning to the corpus question, I'm weighing a few candidates by line length, indentation, and how words are separated: Python code, English Wikipedia, and romanized Japanese text.

I'm weighing the romaji candidate for the corpus, since letters like l, q, v, and x would barely appear there. The prev-token head L0H2 reads token identity, and its singular values fall off gradually, so I'm checking whether several letters are effectively unused, which would shrink the effective rank of its input subspace.

Next I'm weighing whether the newline-token mechanism could be what separates the candidate corpora. Newlines are rare in English prose, far more frequent in Python, and in romanized text they would fall at paragraph breaks, so the S-channel heads copying key-token position might behave quite differently across them.

Now I'm looking at whether the newline token drives the segmentation: in Python it would matter for indentation and line-level structure, while in English prose it would show up far less often.

I'm still working out how the score scales with the query and key projections, and whether I can recover the token embeddings from the layer-0 structure. The embeddings aren't directly visible, so I'm weighing whether MLP0 could help pin them down.

I'm weighing whether the romaji candidate could be told apart by its phonotactics: after a consonant the next character is nearly always a vowel, except for n, the doubled-consonant cases, and the y and tch forms. That would make the previous-token head especially important there, just as it is for English. I'm also wondering whether the number of distinct characters the previous-token head has to separate, around 24 for romaji against about 28 for English or Python, would show up in how many dimensions its copied identity actually uses.

I'm checking whether the residual dimensions with large values, combined with the LayerNorm weights, could reveal anything about the token embeddings, though I haven't settled on a method yet.

I'm noting that some residual dimensions have very small LayerNorm weights, so layer-0 attention largely ignores them while MLP0 relies on them, which makes me wonder what they encode. I also think the embedding could be spread across all dimensions rather than concentrated where attention reads. Separately, I'm looking at the positional eigen-angles of L0H2, which span a wide range of values.

I'm wondering whether the positional embedding carries any corpus signal, such as a period-4 pattern from Python indentation, but since training windows start at random offsets, absolute positions shouldn't reveal the corpus. I'm also returning to whether L0H0 reads the query position against special key tokens like spaces, which would let the heads locate word starts, though a rank-1 QK with no recency term might just average over every space in the window.

I'm weighing whether the broad heads L0H0, L0H1, L0H4, and L0H5 just average over key content, and whether their recency-like position term sharpens attention as the window grows. I'm also turning to the corpus question, since the training budget of about one billion characters, split across Python, English, and romanized Japanese, could help narrow it down.

I'm considering how Python's line structure would look after normalization: leading indentation survives, while punctuation-only lines and blank lines collapse, and digits turn into spaces, which would make the corpus easy to tell apart from prose.

I'm thinking about how after a newline, the indentation could be predicted by copying the previous line's indentation, which suggests a head that attends to the same column in the prior line. I'm then checking whether the L1 heads read the S channel on the key side, and what the query side reads from positional, token, or S subspaces.

Now I'm breaking down the L1 heads' QK circuit by subspace: positional from L0H2, token input from its OV, the S channel, the previous-token output, and the remainder. I'm measuring how much of the query and key energy falls into each block.

